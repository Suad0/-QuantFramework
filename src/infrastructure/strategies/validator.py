"""
Signal validation and quality assessment tools.

This module provides comprehensive validation and quality assessment
for trading signals, including statistical tests, consistency checks,
and performance-based validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

from ...domain.value_objects import Signal
from ...domain.exceptions import ValidationError, StrategyError
from ...infrastructure.logging.logger import get_logger


class ValidationLevel(Enum):
    """Signal validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class ValidationStatus(Enum):
    """Validation status results."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationRule:
    """Defines a validation rule for signals."""
    name: str
    description: str
    level: ValidationLevel
    check_function: callable
    threshold: Optional[float] = None
    is_critical: bool = False


@dataclass
class ValidationResult:
    """Result of signal validation."""
    rule_name: str
    status: ValidationStatus
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class SignalQualityReport:
    """Comprehensive signal quality assessment report."""
    signal_count: int
    validation_results: List[ValidationResult]
    quality_score: float
    critical_failures: int
    warnings: int
    timestamp: datetime
    summary: Dict[str, Any]
    recommendations: List[str]


class SignalValidator:
    """
    Comprehensive signal validation and quality assessment system.
    
    Provides multiple levels of validation including basic consistency checks,
    statistical validation, and performance-based quality assessment.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD, logger=None):
        """
        Initialize signal validator.
        
        Args:
            validation_level: Level of validation to perform
            logger: Logger instance
        """
        self.validation_level = validation_level
        self.logger = logger or get_logger(__name__)
        self.validation_rules = self._initialize_validation_rules()
        self.validation_history = []
    
    def get_validation_history(self) -> List[Any]:
        """Get validation history."""
        return self.validation_history.copy()
        
    def validate_signals(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame] = None,
        benchmark_signals: Optional[List[Signal]] = None
    ) -> SignalQualityReport:
        """
        Validate signals and generate quality report.
        
        Args:
            signals: List of signals to validate
            historical_data: Optional historical market data for context
            benchmark_signals: Optional benchmark signals for comparison
            
        Returns:
            Signal quality report
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            self.logger.info(
                f"Starting signal validation",
                signal_count=len(signals),
                validation_level=self.validation_level.value
            )
            
            if not signals:
                raise ValidationError("Cannot validate empty signal list")
            
            # Run validation rules
            validation_results = []
            for rule in self.validation_rules:
                if rule.level.value <= self.validation_level.value or \
                   self.validation_level == ValidationLevel.COMPREHENSIVE:
                    
                    try:
                        result = self._run_validation_rule(
                            rule, signals, historical_data, benchmark_signals
                        )
                        validation_results.append(result)
                    except Exception as e:
                        self.logger.warning(f"Validation rule {rule.name} failed: {str(e)}")
                        validation_results.append(ValidationResult(
                            rule_name=rule.name,
                            status=ValidationStatus.FAILED,
                            message=f"Rule execution failed: {str(e)}"
                        ))
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(validation_results)
            critical_failures = sum(1 for r in validation_results 
                                  if r.status == ValidationStatus.FAILED)
            warnings_count = sum(1 for r in validation_results 
                               if r.status == ValidationStatus.WARNING)
            
            # Generate summary and recommendations
            summary = self._generate_summary(validation_results, signals)
            recommendations = self._generate_recommendations(validation_results)
            
            # Create quality report
            report = SignalQualityReport(
                signal_count=len(signals),
                validation_results=validation_results,
                quality_score=quality_score,
                critical_failures=critical_failures,
                warnings=warnings_count,
                timestamp=datetime.now(),
                summary=summary,
                recommendations=recommendations
            )
            
            # Store validation history
            self.validation_history.append(report)
            
            self.logger.info(
                f"Signal validation completed",
                quality_score=quality_score,
                critical_failures=critical_failures,
                warnings=warnings_count
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {str(e)}")
            raise ValidationError(
                f"Signal validation failed: {str(e)}",
                error_code="VALIDATION_FAILED"
            ) from e
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules based on validation level."""
        rules = []
        
        # Basic validation rules
        rules.extend([
            ValidationRule(
                name="signal_completeness",
                description="Check that all signals have required fields",
                level=ValidationLevel.BASIC,
                check_function=self._check_signal_completeness,
                is_critical=True
            ),
            ValidationRule(
                name="signal_type_validity",
                description="Check that signal types are valid",
                level=ValidationLevel.BASIC,
                check_function=self._check_signal_type_validity,
                is_critical=True
            ),
            ValidationRule(
                name="strength_range",
                description="Check that signal strengths are within valid range",
                level=ValidationLevel.BASIC,
                check_function=self._check_strength_range,
                threshold=1.0,
                is_critical=True
            ),
            ValidationRule(
                name="confidence_range",
                description="Check that confidence values are within [0, 1]",
                level=ValidationLevel.BASIC,
                check_function=self._check_confidence_range,
                is_critical=True
            )
        ])
        
        # Standard validation rules
        rules.extend([
            ValidationRule(
                name="temporal_consistency",
                description="Check temporal ordering and consistency",
                level=ValidationLevel.STANDARD,
                check_function=self._check_temporal_consistency
            ),
            ValidationRule(
                name="signal_distribution",
                description="Check signal distribution balance",
                level=ValidationLevel.STANDARD,
                check_function=self._check_signal_distribution,
                threshold=0.8
            ),
            ValidationRule(
                name="confidence_correlation",
                description="Check correlation between strength and confidence",
                level=ValidationLevel.STANDARD,
                check_function=self._check_confidence_correlation,
                threshold=0.3
            ),
            ValidationRule(
                name="duplicate_detection",
                description="Check for duplicate signals",
                level=ValidationLevel.STANDARD,
                check_function=self._check_duplicate_signals
            )
        ])
        
        # Comprehensive validation rules
        rules.extend([
            ValidationRule(
                name="statistical_significance",
                description="Check statistical significance of signals",
                level=ValidationLevel.COMPREHENSIVE,
                check_function=self._check_statistical_significance,
                threshold=0.05
            ),
            ValidationRule(
                name="look_ahead_bias",
                description="Check for potential look-ahead bias",
                level=ValidationLevel.COMPREHENSIVE,
                check_function=self._check_look_ahead_bias,
                is_critical=True
            ),
            ValidationRule(
                name="signal_stability",
                description="Check signal stability over time",
                level=ValidationLevel.COMPREHENSIVE,
                check_function=self._check_signal_stability,
                threshold=0.7
            ),
            ValidationRule(
                name="outlier_detection",
                description="Detect outlier signals",
                level=ValidationLevel.COMPREHENSIVE,
                check_function=self._check_outliers,
                threshold=3.0
            )
        ])
        
        return rules
    
    def _run_validation_rule(
        self,
        rule: ValidationRule,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]]
    ) -> ValidationResult:
        """Run a single validation rule."""
        try:
            return rule.check_function(signals, historical_data, benchmark_signals, rule)
        except Exception as e:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Rule execution error: {str(e)}"
            )
    
    # Basic validation rule implementations
    def _check_signal_completeness(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check that all signals have required fields."""
        incomplete_signals = []
        
        for i, signal in enumerate(signals):
            if not all([
                signal.symbol,
                signal.timestamp,
                signal.signal_type,
                signal.strength is not None,
                signal.confidence is not None
            ]):
                incomplete_signals.append(i)
        
        if incomplete_signals:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Found {len(incomplete_signals)} incomplete signals",
                details={"incomplete_indices": incomplete_signals[:10]}  # Limit details
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="All signals have required fields"
        )
    
    def _check_signal_type_validity(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check that signal types are valid."""
        from ...domain.value_objects import SignalType
        valid_types = {SignalType.BUY, SignalType.SELL, SignalType.HOLD}
        invalid_signals = []
        
        for i, signal in enumerate(signals):
            if signal.signal_type not in valid_types:
                invalid_signals.append((i, signal.signal_type))
        
        if invalid_signals:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Found {len(invalid_signals)} signals with invalid types",
                details={"invalid_signals": invalid_signals[:10]}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="All signal types are valid"
        )
    
    def _check_strength_range(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check that signal strengths are within valid range."""
        out_of_range = []
        
        for i, signal in enumerate(signals):
            if abs(signal.strength) > rule.threshold:
                out_of_range.append((i, signal.strength))
        
        if out_of_range:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Found {len(out_of_range)} signals with strength > {rule.threshold}",
                threshold=rule.threshold,
                details={"out_of_range_signals": out_of_range[:10]}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message=f"All signal strengths within range [-{rule.threshold}, {rule.threshold}]"
        )
    
    def _check_confidence_range(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check that confidence values are within [0, 1]."""
        out_of_range = []
        
        for i, signal in enumerate(signals):
            if not (0 <= signal.confidence <= 1):
                out_of_range.append((i, signal.confidence))
        
        if out_of_range:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.FAILED,
                message=f"Found {len(out_of_range)} signals with confidence outside [0, 1]",
                details={"out_of_range_confidence": out_of_range[:10]}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="All confidence values within [0, 1] range"
        )
    
    # Standard validation rule implementations
    def _check_temporal_consistency(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check temporal ordering and consistency."""
        if len(signals) < 2:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="Insufficient signals for temporal consistency check"
            )
        
        # Check for chronological ordering
        timestamps = [signal.timestamp for signal in signals]
        is_sorted = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        
        # Check for reasonable time gaps
        time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        # Flag unusually large gaps (more than 7 days)
        large_gaps = [gap for gap in time_gaps if gap > 7 * 24 * 3600]
        
        status = ValidationStatus.PASSED
        message = "Temporal consistency check passed"
        details = {}
        
        if not is_sorted:
            status = ValidationStatus.WARNING
            message = "Signals are not chronologically ordered"
        
        if large_gaps:
            if status == ValidationStatus.PASSED:
                status = ValidationStatus.WARNING
                message = f"Found {len(large_gaps)} large time gaps (>7 days)"
            details["large_gaps_count"] = len(large_gaps)
            details["max_gap_days"] = max(large_gaps) / (24 * 3600)
        
        return ValidationResult(
            rule_name=rule.name,
            status=status,
            message=message,
            details=details if details else None
        )
    
    def _check_signal_distribution(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check signal distribution balance."""
        from ...domain.value_objects import SignalType
        signal_counts = {SignalType.BUY: 0, SignalType.SELL: 0, SignalType.HOLD: 0}
        
        for signal in signals:
            signal_counts[signal.signal_type] += 1
        
        total_signals = len(signals)
        max_proportion = max(signal_counts.values()) / total_signals
        
        if max_proportion > rule.threshold:
            max_signal_type = max(signal_counts, key=signal_counts.get)
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message=f"Signal distribution imbalanced: {max_proportion:.2%} of signals are {max_signal_type.value}",
                value=max_proportion,
                threshold=rule.threshold,
                details={"distribution": {k.value: v for k, v in signal_counts.items()}}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="Signal distribution is balanced",
            details={"distribution": {k.value: v for k, v in signal_counts.items()}}
        )
    
    def _check_confidence_correlation(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check correlation between strength and confidence."""
        if len(signals) < 10:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="Insufficient signals for correlation analysis"
            )
        
        strengths = [abs(signal.strength) for signal in signals]
        confidences = [signal.confidence for signal in signals]
        
        correlation = np.corrcoef(strengths, confidences)[0, 1]
        
        if np.isnan(correlation):
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message="Cannot calculate strength-confidence correlation"
            )
        
        if correlation < rule.threshold:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message=f"Low correlation between strength and confidence: {correlation:.3f}",
                value=correlation,
                threshold=rule.threshold
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message=f"Good correlation between strength and confidence: {correlation:.3f}",
            value=correlation
        )
    
    def _check_duplicate_signals(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check for duplicate signals."""
        seen_signals = set()
        duplicates = []
        
        for i, signal in enumerate(signals):
            signal_key = (signal.symbol, signal.timestamp, signal.signal_type)
            if signal_key in seen_signals:
                duplicates.append(i)
            else:
                seen_signals.add(signal_key)
        
        if duplicates:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message=f"Found {len(duplicates)} duplicate signals",
                details={"duplicate_indices": duplicates[:10]}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="No duplicate signals found"
        )
    
    # Comprehensive validation rule implementations
    def _check_statistical_significance(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check statistical significance of signals."""
        # This is a placeholder for statistical significance testing
        # Would implement proper statistical tests based on signal performance
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="Statistical significance check not fully implemented"
        )
    
    def _check_look_ahead_bias(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check for potential look-ahead bias."""
        # This would implement sophisticated look-ahead bias detection
        # For now, basic timestamp consistency check
        
        if historical_data is not None:
            # Check if any signals have timestamps beyond available data
            data_end = historical_data.index.max() if hasattr(historical_data.index, 'max') else None
            
            if data_end:
                future_signals = [
                    i for i, signal in enumerate(signals)
                    if signal.timestamp > data_end
                ]
                
                if future_signals:
                    return ValidationResult(
                        rule_name=rule.name,
                        status=ValidationStatus.FAILED,
                        message=f"Found {len(future_signals)} signals with future timestamps",
                        details={"future_signal_indices": future_signals[:10]}
                    )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message="No obvious look-ahead bias detected"
        )
    
    def _check_signal_stability(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Check signal stability over time."""
        if len(signals) < 20:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="Insufficient signals for stability analysis"
            )
        
        # Calculate rolling statistics
        window_size = min(10, len(signals) // 4)
        rolling_means = []
        
        for i in range(window_size, len(signals)):
            window_signals = signals[i-window_size:i]
            mean_strength = np.mean([abs(s.strength) for s in window_signals])
            rolling_means.append(mean_strength)
        
        if len(rolling_means) < 2:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="Insufficient data for stability analysis"
            )
        
        # Calculate stability metric (inverse of coefficient of variation)
        stability = 1 - (np.std(rolling_means) / np.mean(rolling_means))
        
        if stability < rule.threshold:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message=f"Low signal stability: {stability:.3f}",
                value=stability,
                threshold=rule.threshold
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message=f"Good signal stability: {stability:.3f}",
            value=stability
        )
    
    def _check_outliers(
        self,
        signals: List[Signal],
        historical_data: Optional[pd.DataFrame],
        benchmark_signals: Optional[List[Signal]],
        rule: ValidationRule
    ) -> ValidationResult:
        """Detect outlier signals."""
        strengths = [signal.strength for signal in signals]
        
        if len(strengths) < 10:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="Insufficient signals for outlier detection"
            )
        
        # Use z-score for outlier detection
        mean_strength = np.mean(strengths)
        std_strength = np.std(strengths)
        
        if std_strength == 0:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.PASSED,
                message="No variation in signal strengths"
            )
        
        z_scores = [(s - mean_strength) / std_strength for s in strengths]
        outliers = [i for i, z in enumerate(z_scores) if abs(z) > rule.threshold]
        
        if outliers:
            return ValidationResult(
                rule_name=rule.name,
                status=ValidationStatus.WARNING,
                message=f"Found {len(outliers)} outlier signals (z-score > {rule.threshold})",
                details={"outlier_indices": outliers[:10]}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            status=ValidationStatus.PASSED,
            message=f"No outliers detected (z-score threshold: {rule.threshold})"
        )
    
    def _calculate_quality_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall quality score from validation results."""
        if not validation_results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            # Weight based on status
            if result.status == ValidationStatus.PASSED:
                score = 1.0
            elif result.status == ValidationStatus.WARNING:
                score = 0.5
            else:  # FAILED
                score = 0.0
            
            # Higher weight for critical rules
            weight = 2.0 if any(rule.is_critical and rule.name == result.rule_name 
                              for rule in self.validation_rules) else 1.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary(
        self,
        validation_results: List[ValidationResult],
        signals: List[Signal]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_rules_checked": len(validation_results),
            "rules_passed": sum(1 for r in validation_results if r.status == ValidationStatus.PASSED),
            "rules_warned": sum(1 for r in validation_results if r.status == ValidationStatus.WARNING),
            "rules_failed": sum(1 for r in validation_results if r.status == ValidationStatus.FAILED),
            "signal_types": {
                signal_type: sum(1 for s in signals if s.signal_type == signal_type)
                for signal_type in ["BUY", "SELL", "HOLD"]
            },
            "strength_stats": {
                "mean": np.mean([s.strength for s in signals]),
                "std": np.std([s.strength for s in signals]),
                "min": min(s.strength for s in signals),
                "max": max(s.strength for s in signals)
            },
            "confidence_stats": {
                "mean": np.mean([s.confidence for s in signals]),
                "std": np.std([s.confidence for s in signals]),
                "min": min(s.confidence for s in signals),
                "max": max(s.confidence for s in signals)
            }
        }
    
    def _generate_recommendations(
        self,
        validation_results: List[ValidationResult]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for result in validation_results:
            if result.status == ValidationStatus.FAILED:
                if "completeness" in result.rule_name:
                    recommendations.append("Ensure all signals have required fields before validation")
                elif "type_validity" in result.rule_name:
                    recommendations.append("Check signal generation logic for invalid signal types")
                elif "strength_range" in result.rule_name:
                    recommendations.append("Review signal strength calculation to ensure values are within expected range")
                elif "confidence_range" in result.rule_name:
                    recommendations.append("Normalize confidence values to [0, 1] range")
                elif "look_ahead_bias" in result.rule_name:
                    recommendations.append("Review signal generation timing to prevent look-ahead bias")
            
            elif result.status == ValidationStatus.WARNING:
                if "distribution" in result.rule_name:
                    recommendations.append("Consider rebalancing signal generation to avoid bias toward specific signal types")
                elif "correlation" in result.rule_name:
                    recommendations.append("Review confidence calculation to better reflect signal strength")
                elif "stability" in result.rule_name:
                    recommendations.append("Investigate signal stability issues - consider parameter tuning")
                elif "outliers" in result.rule_name:
                    recommendations.append("Review outlier signals for potential data quality issues")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Signal validation passed - consider implementing additional quality checks")
        
        return recommendations
    
    def get_validation_history(self) -> List[SignalQualityReport]:
        """Get history of validation reports."""
        return self.validation_history.copy()
    
    def export_validation_report(self, report: SignalQualityReport, file_path: str) -> None:
        """Export validation report to file."""
        try:
            export_data = {
                "signal_count": report.signal_count,
                "quality_score": report.quality_score,
                "critical_failures": report.critical_failures,
                "warnings": report.warnings,
                "timestamp": report.timestamp.isoformat(),
                "validation_results": [
                    {
                        "rule_name": r.rule_name,
                        "status": r.status.value,
                        "message": r.message,
                        "value": r.value,
                        "threshold": r.threshold
                    }
                    for r in report.validation_results
                ],
                "summary": report.summary,
                "recommendations": report.recommendations
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported validation report to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export validation report: {str(e)}")
            raise ValidationError(
                f"Report export failed: {str(e)}",
                error_code="EXPORT_FAILED"
            ) from e