"""
Data quality validation system with configurable rules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a data quality issue."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    is_valid: bool
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """Check if report contains errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)


@dataclass
class ValidationRule:
    """Configuration for a validation rule."""
    name: str
    description: str
    severity: ValidationSeverity
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    validator_func: Optional[Callable] = None


class DataQualityValidator:
    """Comprehensive data quality validation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_default_rules()
        
        # Load custom rules from config
        if 'custom_rules' in self.config:
            self._load_custom_rules(self.config['custom_rules'])
    
    def _initialize_default_rules(self) -> Dict[str, ValidationRule]:
        """Initialize default validation rules."""
        return {
            'missing_values': ValidationRule(
                name='missing_values',
                description='Check for missing values in critical columns',
                severity=ValidationSeverity.WARNING,
                parameters={'max_missing_pct': 0.05, 'critical_columns': ['close', 'volume']},
                validator_func=self._validate_missing_values
            ),
            'duplicate_records': ValidationRule(
                name='duplicate_records',
                description='Check for duplicate records',
                severity=ValidationSeverity.ERROR,
                validator_func=self._validate_duplicates
            ),
            'price_outliers': ValidationRule(
                name='price_outliers',
                description='Detect price outliers using statistical methods',
                severity=ValidationSeverity.WARNING,
                parameters={'z_threshold': 3.0, 'price_columns': ['open', 'high', 'low', 'close']},
                validator_func=self._validate_price_outliers
            ),
            'volume_outliers': ValidationRule(
                name='volume_outliers',
                description='Detect volume outliers',
                severity=ValidationSeverity.WARNING,
                parameters={'z_threshold': 4.0},
                validator_func=self._validate_volume_outliers
            ),
            'price_consistency': ValidationRule(
                name='price_consistency',
                description='Validate OHLC price relationships',
                severity=ValidationSeverity.ERROR,
                validator_func=self._validate_price_consistency
            ),
            'date_continuity': ValidationRule(
                name='date_continuity',
                description='Check for gaps in date series',
                severity=ValidationSeverity.WARNING,
                parameters={'max_gap_days': 7, 'exclude_weekends': True},
                validator_func=self._validate_date_continuity
            ),
            'negative_prices': ValidationRule(
                name='negative_prices',
                description='Check for negative prices',
                severity=ValidationSeverity.CRITICAL,
                validator_func=self._validate_negative_prices
            ),
            'zero_volume': ValidationRule(
                name='zero_volume',
                description='Check for zero volume trading days',
                severity=ValidationSeverity.INFO,
                parameters={'max_zero_volume_pct': 0.01},
                validator_func=self._validate_zero_volume
            ),
            'data_freshness': ValidationRule(
                name='data_freshness',
                description='Check data freshness',
                severity=ValidationSeverity.WARNING,
                parameters={'max_age_days': 2},
                validator_func=self._validate_data_freshness
            ),
            'corporate_actions': ValidationRule(
                name='corporate_actions',
                description='Detect potential corporate actions',
                severity=ValidationSeverity.INFO,
                parameters={'split_threshold': 0.5, 'dividend_threshold': 0.1},
                validator_func=self._validate_corporate_actions
            )
        }
    
    def _load_custom_rules(self, custom_rules: List[Dict[str, Any]]) -> None:
        """Load custom validation rules from configuration."""
        for rule_config in custom_rules:
            rule = ValidationRule(
                name=rule_config['name'],
                description=rule_config.get('description', ''),
                severity=ValidationSeverity(rule_config.get('severity', 'warning')),
                enabled=rule_config.get('enabled', True),
                parameters=rule_config.get('parameters', {})
            )
            self.rules[rule.name] = rule
    
    async def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data quality validation."""
        if data.empty:
            return DataQualityReport(
                is_valid=False,
                total_rows=0,
                total_columns=0,
                issues=[ValidationIssue(
                    rule_name='empty_data',
                    severity=ValidationSeverity.CRITICAL,
                    message='Dataset is empty'
                )]
            )
        
        report = DataQualityReport(
            is_valid=True,
            total_rows=len(data),
            total_columns=len(data.columns)
        )
        
        # Run all enabled validation rules
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                if rule.validator_func:
                    issues = rule.validator_func(data, rule)
                    report.issues.extend(issues)
                
            except Exception as e:
                self.logger.error(f"Validation rule '{rule_name}' failed: {str(e)}")
                report.issues.append(ValidationIssue(
                    rule_name=rule_name,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule failed: {str(e)}"
                ))
        
        # Determine overall validity
        report.is_valid = not (report.has_critical_issues() or report.has_errors())
        
        # Generate summary
        report.summary = self._generate_summary(report)
        
        self.logger.info(f"Data quality validation completed. Valid: {report.is_valid}, Issues: {len(report.issues)}")
        
        return report
    
    def _validate_missing_values(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate missing values in data."""
        issues = []
        max_missing_pct = rule.parameters.get('max_missing_pct', 0.05)
        critical_columns = rule.parameters.get('critical_columns', [])
        
        for column in data.columns:
            if column in ['symbol', 'provider', 'timestamp']:
                continue
                
            missing_count = data[column].isna().sum()
            missing_pct = missing_count / len(data)
            
            if missing_pct > max_missing_pct:
                severity = ValidationSeverity.ERROR if column in critical_columns else rule.severity
                
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    severity=severity,
                    message=f"Column '{column}' has {missing_pct:.2%} missing values (threshold: {max_missing_pct:.2%})",
                    affected_columns=[column],
                    metadata={'missing_count': missing_count, 'missing_pct': missing_pct}
                ))
        
        return issues
    
    def _validate_duplicates(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate duplicate records."""
        issues = []
        
        # Check for duplicates based on symbol and date
        if 'symbol' in data.columns and 'date' in data.columns:
            duplicate_mask = data.duplicated(subset=['symbol', 'date'], keep=False)
            duplicate_count = duplicate_mask.sum()
            
            if duplicate_count > 0:
                duplicate_rows = data.index[duplicate_mask].tolist()
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"Found {duplicate_count} duplicate records",
                    affected_rows=duplicate_rows,
                    metadata={'duplicate_count': duplicate_count}
                ))
        
        return issues
    
    def _validate_price_outliers(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate price outliers using z-score."""
        issues = []
        z_threshold = rule.parameters.get('z_threshold', 3.0)
        price_columns = rule.parameters.get('price_columns', ['open', 'high', 'low', 'close'])
        
        for column in price_columns:
            if column not in data.columns:
                continue
            
            # Calculate z-scores for each symbol separately
            if 'symbol' in data.columns:
                for symbol in data['symbol'].unique():
                    symbol_data = data[data['symbol'] == symbol]
                    if len(symbol_data) < 10:  # Need sufficient data for z-score
                        continue
                    
                    z_scores = np.abs((symbol_data[column] - symbol_data[column].mean()) / symbol_data[column].std())
                    outlier_mask = z_scores > z_threshold
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        outlier_rows = symbol_data.index[outlier_mask].tolist()
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Found {outlier_count} price outliers in {column} for {symbol}",
                            affected_rows=outlier_rows,
                            affected_columns=[column],
                            metadata={'symbol': symbol, 'outlier_count': outlier_count, 'z_threshold': z_threshold}
                        ))
            else:
                # Global outlier detection
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outlier_mask = z_scores > z_threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outlier_rows = data.index[outlier_mask].tolist()
                    issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {outlier_count} price outliers in {column}",
                        affected_rows=outlier_rows,
                        affected_columns=[column],
                        metadata={'outlier_count': outlier_count, 'z_threshold': z_threshold}
                    ))
        
        return issues
    
    def _validate_volume_outliers(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate volume outliers."""
        issues = []
        
        if 'volume' not in data.columns:
            return issues
        
        z_threshold = rule.parameters.get('z_threshold', 4.0)
        
        # Volume outliers by symbol
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol]
                if len(symbol_data) < 10:
                    continue
                
                # Use log transformation for volume to handle skewness
                log_volume = np.log1p(symbol_data['volume'])
                z_scores = np.abs((log_volume - log_volume.mean()) / log_volume.std())
                outlier_mask = z_scores > z_threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    outlier_rows = symbol_data.index[outlier_mask].tolist()
                    issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Found {outlier_count} volume outliers for {symbol}",
                        affected_rows=outlier_rows,
                        affected_columns=['volume'],
                        metadata={'symbol': symbol, 'outlier_count': outlier_count}
                    ))
        
        return issues
    
    def _validate_price_consistency(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate OHLC price relationships."""
        issues = []
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return issues
        
        # Check High >= max(Open, Close) and Low <= min(Open, Close)
        invalid_high = data['high'] < data[['open', 'close']].max(axis=1)
        invalid_low = data['low'] > data[['open', 'close']].min(axis=1)
        
        invalid_high_count = invalid_high.sum()
        invalid_low_count = invalid_low.sum()
        
        if invalid_high_count > 0:
            invalid_rows = data.index[invalid_high].tolist()
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Found {invalid_high_count} records where High < max(Open, Close)",
                affected_rows=invalid_rows,
                affected_columns=['high', 'open', 'close'],
                metadata={'invalid_count': invalid_high_count}
            ))
        
        if invalid_low_count > 0:
            invalid_rows = data.index[invalid_low].tolist()
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Found {invalid_low_count} records where Low > min(Open, Close)",
                affected_rows=invalid_rows,
                affected_columns=['low', 'open', 'close'],
                metadata={'invalid_count': invalid_low_count}
            ))
        
        return issues
    
    def _validate_date_continuity(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate date continuity."""
        issues = []
        
        if 'date' not in data.columns:
            return issues
        
        max_gap_days = rule.parameters.get('max_gap_days', 7)
        exclude_weekends = rule.parameters.get('exclude_weekends', True)
        
        # Check date gaps by symbol
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].sort_values('date')
                if len(symbol_data) < 2:
                    continue
                
                dates = pd.to_datetime(symbol_data['date'])
                date_diffs = dates.diff().dt.days
                
                if exclude_weekends:
                    # Adjust for weekends (business days)
                    business_days = pd.bdate_range(dates.min(), dates.max())
                    expected_days = len(business_days) - 1
                    actual_days = len(dates) - 1
                    
                    if expected_days - actual_days > max_gap_days:
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Date gap detected for {symbol}: missing {expected_days - actual_days} business days",
                            metadata={'symbol': symbol, 'missing_days': expected_days - actual_days}
                        ))
                else:
                    large_gaps = date_diffs > max_gap_days
                    gap_count = large_gaps.sum()
                    
                    if gap_count > 0:
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Found {gap_count} date gaps > {max_gap_days} days for {symbol}",
                            metadata={'symbol': symbol, 'gap_count': gap_count}
                        ))
        
        return issues
    
    def _validate_negative_prices(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate for negative prices."""
        issues = []
        
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        
        for column in price_columns:
            if column not in data.columns:
                continue
            
            negative_mask = data[column] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                negative_rows = data.index[negative_mask].tolist()
                issues.append(ValidationIssue(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=f"Found {negative_count} negative values in {column}",
                    affected_rows=negative_rows,
                    affected_columns=[column],
                    metadata={'negative_count': negative_count}
                ))
        
        return issues
    
    def _validate_zero_volume(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate zero volume trading days."""
        issues = []
        
        if 'volume' not in data.columns:
            return issues
        
        max_zero_volume_pct = rule.parameters.get('max_zero_volume_pct', 0.01)
        
        zero_volume_mask = data['volume'] == 0
        zero_volume_count = zero_volume_mask.sum()
        zero_volume_pct = zero_volume_count / len(data)
        
        if zero_volume_pct > max_zero_volume_pct:
            zero_volume_rows = data.index[zero_volume_mask].tolist()
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Found {zero_volume_pct:.2%} zero volume days (threshold: {max_zero_volume_pct:.2%})",
                affected_rows=zero_volume_rows,
                affected_columns=['volume'],
                metadata={'zero_volume_count': zero_volume_count, 'zero_volume_pct': zero_volume_pct}
            ))
        
        return issues
    
    def _validate_data_freshness(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Validate data freshness."""
        issues = []
        
        if 'date' not in data.columns:
            return issues
        
        max_age_days = rule.parameters.get('max_age_days', 2)
        
        latest_date = pd.to_datetime(data['date']).max()
        current_date = pd.Timestamp.now().normalize()
        age_days = (current_date - latest_date).days
        
        if age_days > max_age_days:
            issues.append(ValidationIssue(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"Data is {age_days} days old (threshold: {max_age_days} days)",
                metadata={'age_days': age_days, 'latest_date': latest_date.isoformat()}
            ))
        
        return issues
    
    def _validate_corporate_actions(self, data: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Detect potential corporate actions."""
        issues = []
        
        if not all(col in data.columns for col in ['close', 'adj_close']):
            return issues
        
        split_threshold = rule.parameters.get('split_threshold', 0.5)
        dividend_threshold = rule.parameters.get('dividend_threshold', 0.1)
        
        # Detect potential stock splits (large price drops)
        if 'symbol' in data.columns:
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].sort_values('date')
                if len(symbol_data) < 2:
                    continue
                
                price_changes = symbol_data['close'].pct_change()
                
                # Potential splits (large negative price changes)
                potential_splits = price_changes < -split_threshold
                split_count = potential_splits.sum()
                
                if split_count > 0:
                    split_rows = symbol_data.index[potential_splits].tolist()
                    issues.append(ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=f"Detected {split_count} potential stock splits for {symbol}",
                        affected_rows=split_rows,
                        metadata={'symbol': symbol, 'split_count': split_count, 'type': 'split'}
                    ))
                
                # Potential dividends (difference between close and adj_close)
                if 'adj_close' in symbol_data.columns:
                    dividend_impact = (symbol_data['close'] - symbol_data['adj_close']) / symbol_data['close']
                    potential_dividends = dividend_impact > dividend_threshold
                    dividend_count = potential_dividends.sum()
                    
                    if dividend_count > 0:
                        dividend_rows = symbol_data.index[potential_dividends].tolist()
                        issues.append(ValidationIssue(
                            rule_name=rule.name,
                            severity=rule.severity,
                            message=f"Detected {dividend_count} potential dividend payments for {symbol}",
                            affected_rows=dividend_rows,
                            metadata={'symbol': symbol, 'dividend_count': dividend_count, 'type': 'dividend'}
                        ))
        
        return issues
    
    def _generate_summary(self, report: DataQualityReport) -> Dict[str, Any]:
        """Generate summary statistics for the quality report."""
        summary = {
            'total_issues': len(report.issues),
            'issues_by_severity': {
                'critical': len(report.get_issues_by_severity(ValidationSeverity.CRITICAL)),
                'error': len(report.get_issues_by_severity(ValidationSeverity.ERROR)),
                'warning': len(report.get_issues_by_severity(ValidationSeverity.WARNING)),
                'info': len(report.get_issues_by_severity(ValidationSeverity.INFO))
            },
            'rules_executed': len([rule for rule in self.rules.values() if rule.enabled]),
            'data_quality_score': self._calculate_quality_score(report)
        }
        
        return summary
    
    def _calculate_quality_score(self, report: DataQualityReport) -> float:
        """Calculate overall data quality score (0-100)."""
        if report.total_rows == 0:
            return 0.0
        
        # Weight different severity levels
        weights = {
            ValidationSeverity.CRITICAL: 10,
            ValidationSeverity.ERROR: 5,
            ValidationSeverity.WARNING: 2,
            ValidationSeverity.INFO: 1
        }
        
        total_penalty = sum(
            len(report.get_issues_by_severity(severity)) * weight
            for severity, weight in weights.items()
        )
        
        # Normalize by data size
        max_possible_penalty = report.total_rows * sum(weights.values())
        
        if max_possible_penalty == 0:
            return 100.0
        
        score = max(0, 100 - (total_penalty / max_possible_penalty * 100))
        return round(score, 2)
    
    def add_custom_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule."""
        self.rules[rule.name] = rule
    
    def enable_rule(self, rule_name: str) -> None:
        """Enable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
    
    def disable_rule(self, rule_name: str) -> None:
        """Disable a validation rule."""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
    
    def get_enabled_rules(self) -> List[str]:
        """Get list of enabled rule names."""
        return [name for name, rule in self.rules.items() if rule.enabled]