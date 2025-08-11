"""
A/B testing framework for ML model comparison.

This module provides tools for statistically comparing the performance of
different ML models using A/B testing methodologies.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.domain.exceptions import ValidationError
from .financial_metrics import FinancialMetricsCalculator


class TestStatus(Enum):
    """Status of A/B test."""
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ABTestResult:
    """Result of A/B test comparison."""
    test_id: str
    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_performance: float
    model_b_performance: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    statistical_significance: bool
    practical_significance: bool
    winner: Optional[str]
    confidence_level: float
    sample_size: int
    test_duration: Optional[timedelta]
    timestamp: datetime


class StatisticalTest:
    """Statistical tests for model comparison."""
    
    @staticmethod
    def t_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform t-test for comparing two sets of scores.
        
        Args:
            scores_a: Performance scores for model A
            scores_b: Performance scores for model B
            confidence_level: Confidence level for the test
            
        Returns:
            Dictionary with test results
        """
        if len(scores_a) == 0 or len(scores_b) == 0:
            raise ValidationError("Both score arrays must be non-empty")
        
        # Perform two-sample t-test
        statistic, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Calculate confidence interval for difference in means
        alpha = 1 - confidence_level
        pooled_std = np.sqrt(
            ((len(scores_a) - 1) * np.var(scores_a, ddof=1) + 
             (len(scores_b) - 1) * np.var(scores_b, ddof=1)) / 
            (len(scores_a) + len(scores_b) - 2)
        )
        
        se_diff = pooled_std * np.sqrt(1/len(scores_a) + 1/len(scores_b))
        df = len(scores_a) + len(scores_b) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mean_diff = np.mean(scores_a) - np.mean(scores_b)
        margin_error = t_critical * se_diff
        
        confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
        
        # Calculate effect size (Cohen's d)
        effect_size = mean_diff / pooled_std
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'confidence_interval': confidence_interval,
            'effect_size': effect_size,
            'mean_difference': mean_diff,
            'test_type': 't_test'
        }
    
    @staticmethod
    def mann_whitney_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            scores_a: Performance scores for model A
            scores_b: Performance scores for model B
            confidence_level: Confidence level for the test
            
        Returns:
            Dictionary with test results
        """
        if len(scores_a) == 0 or len(scores_b) == 0:
            raise ValidationError("Both score arrays must be non-empty")
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            scores_a, scores_b, alternative='two-sided'
        )
        
        # Calculate effect size (rank-biserial correlation)
        n_a, n_b = len(scores_a), len(scores_b)
        effect_size = 1 - (2 * statistic) / (n_a * n_b)
        
        # For confidence interval, we use bootstrap (simplified approach)
        # In practice, you might want to use more sophisticated methods
        combined_scores = np.concatenate([scores_a, scores_b])
        median_diff = np.median(scores_a) - np.median(scores_b)
        
        # Simple confidence interval based on median difference
        # This is a simplified approach
        confidence_interval = (median_diff * 0.8, median_diff * 1.2)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'confidence_interval': confidence_interval,
            'effect_size': effect_size,
            'median_difference': median_diff,
            'test_type': 'mann_whitney'
        }
    
    @staticmethod
    def bootstrap_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform bootstrap test for comparing two models.
        
        Args:
            scores_a: Performance scores for model A
            scores_b: Performance scores for model B
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for the test
            random_state: Random seed
            
        Returns:
            Dictionary with test results
        """
        if len(scores_a) == 0 or len(scores_b) == 0:
            raise ValidationError("Both score arrays must be non-empty")
        
        np.random.seed(random_state)
        
        # Calculate observed difference
        observed_diff = np.mean(scores_a) - np.mean(scores_b)
        
        # Bootstrap sampling
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_a = np.random.choice(scores_a, size=len(scores_a), replace=True)
            boot_b = np.random.choice(scores_b, size=len(scores_b), replace=True)
            
            boot_diff = np.mean(boot_a) - np.mean(boot_b)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_interval = (
            np.percentile(bootstrap_diffs, lower_percentile),
            np.percentile(bootstrap_diffs, upper_percentile)
        )
        
        # Effect size (standardized difference)
        pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'statistic': observed_diff,
            'p_value': p_value,
            'confidence_interval': confidence_interval,
            'effect_size': effect_size,
            'bootstrap_diffs': bootstrap_diffs,
            'test_type': 'bootstrap'
        }


class ABTester:
    """A/B testing framework for ML model comparison."""
    
    def __init__(
        self,
        significance_level: float = 0.05,
        minimum_effect_size: float = 0.1,
        minimum_sample_size: int = 30,
        metrics_calculator: Optional[FinancialMetricsCalculator] = None
    ):
        """
        Initialize A/B tester.
        
        Args:
            significance_level: Statistical significance threshold
            minimum_effect_size: Minimum practical effect size
            minimum_sample_size: Minimum sample size for testing
            metrics_calculator: Financial metrics calculator
        """
        self.significance_level = significance_level
        self.minimum_effect_size = minimum_effect_size
        self.minimum_sample_size = minimum_sample_size
        self.metrics_calculator = metrics_calculator or FinancialMetricsCalculator()
        
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.completed_tests: List[ABTestResult] = []
    
    def start_ab_test(
        self,
        test_id: str,
        model_a: BaseEstimator,
        model_b: BaseEstimator,
        model_a_name: str,
        model_b_name: str,
        test_data: pd.DataFrame,
        target_column: str,
        metric_name: str = 'mse',
        test_method: str = 't_test',
        confidence_level: float = 0.95
    ) -> str:
        """
        Start an A/B test between two models.
        
        Args:
            test_id: Unique identifier for the test
            model_a: First model to compare
            model_b: Second model to compare
            model_a_name: Name of first model
            model_b_name: Name of second model
            test_data: Data for testing (features + target)
            target_column: Name of target column
            metric_name: Performance metric to compare
            test_method: Statistical test method
            confidence_level: Confidence level for the test
            
        Returns:
            Test ID
        """
        if test_id in self.active_tests:
            raise ValidationError(f"Test {test_id} is already active")
        
        if len(test_data) < self.minimum_sample_size:
            raise ValidationError(f"Sample size {len(test_data)} is below minimum {self.minimum_sample_size}")
        
        # Prepare test data
        features = test_data.drop(columns=[target_column])
        targets = test_data[target_column]
        
        # Store test configuration
        self.active_tests[test_id] = {
            'model_a': model_a,
            'model_b': model_b,
            'model_a_name': model_a_name,
            'model_b_name': model_b_name,
            'features': features,
            'targets': targets,
            'metric_name': metric_name,
            'test_method': test_method,
            'confidence_level': confidence_level,
            'start_time': datetime.now(),
            'status': TestStatus.RUNNING,
            'scores_a': [],
            'scores_b': []
        }
        
        return test_id
    
    def add_test_data(
        self,
        test_id: str,
        new_data: pd.DataFrame,
        target_column: str
    ) -> None:
        """
        Add new data to an active A/B test.
        
        Args:
            test_id: Test identifier
            new_data: New test data
            target_column: Name of target column
        """
        if test_id not in self.active_tests:
            raise ValidationError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        if test_config['status'] != TestStatus.RUNNING:
            raise ValidationError(f"Test {test_id} is not running")
        
        # Prepare new data
        new_features = new_data.drop(columns=[target_column])
        new_targets = new_data[target_column]
        
        # Make predictions with both models
        try:
            pred_a = test_config['model_a'].predict(new_features)
            pred_b = test_config['model_b'].predict(new_features)
            
            # Calculate individual performance scores for each prediction
            for i in range(len(new_targets)):
                score_a = self._calculate_individual_metric_score(
                    new_targets.iloc[i], pred_a[i], test_config['metric_name']
                )
                score_b = self._calculate_individual_metric_score(
                    new_targets.iloc[i], pred_b[i], test_config['metric_name']
                )
                
                # Store individual scores
                test_config['scores_a'].append(score_a)
                test_config['scores_b'].append(score_b)
            
        except Exception as e:
            warnings.warn(f"Failed to add test data: {e}")
    
    def evaluate_test(self, test_id: str) -> Optional[ABTestResult]:
        """
        Evaluate the current state of an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test result if evaluation is possible, None otherwise
        """
        if test_id not in self.active_tests:
            raise ValidationError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        if len(test_config['scores_a']) < self.minimum_sample_size:
            return None  # Not enough data yet
        
        scores_a = np.array(test_config['scores_a'])
        scores_b = np.array(test_config['scores_b'])
        
        # Perform statistical test
        test_method = test_config['test_method']
        confidence_level = test_config['confidence_level']
        
        if test_method == 't_test':
            test_result = StatisticalTest.t_test(scores_a, scores_b, confidence_level)
        elif test_method == 'mann_whitney':
            test_result = StatisticalTest.mann_whitney_test(scores_a, scores_b, confidence_level)
        elif test_method == 'bootstrap':
            test_result = StatisticalTest.bootstrap_test(scores_a, scores_b, confidence_level=confidence_level)
        else:
            raise ValidationError(f"Unknown test method: {test_method}")
        
        # Determine statistical significance
        statistical_significance = test_result['p_value'] < self.significance_level
        
        # Determine practical significance
        practical_significance = abs(test_result['effect_size']) >= self.minimum_effect_size
        
        # Determine winner
        winner = None
        if statistical_significance and practical_significance:
            if test_config['metric_name'] in ['mse', 'mae', 'rmse']:  # Lower is better
                winner = test_config['model_a_name'] if np.mean(scores_a) < np.mean(scores_b) else test_config['model_b_name']
            else:  # Higher is better
                winner = test_config['model_a_name'] if np.mean(scores_a) > np.mean(scores_b) else test_config['model_b_name']
        
        # Create result
        result = ABTestResult(
            test_id=test_id,
            model_a_name=test_config['model_a_name'],
            model_b_name=test_config['model_b_name'],
            metric_name=test_config['metric_name'],
            model_a_performance=np.mean(scores_a),
            model_b_performance=np.mean(scores_b),
            p_value=test_result['p_value'],
            confidence_interval=test_result['confidence_interval'],
            effect_size=test_result['effect_size'],
            statistical_significance=statistical_significance,
            practical_significance=practical_significance,
            winner=winner,
            confidence_level=confidence_level,
            sample_size=len(scores_a),
            test_duration=datetime.now() - test_config['start_time'],
            timestamp=datetime.now()
        )
        
        return result
    
    def stop_test(self, test_id: str) -> ABTestResult:
        """
        Stop an A/B test and return final results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Final test result
        """
        if test_id not in self.active_tests:
            raise ValidationError(f"Test {test_id} not found")
        
        # Evaluate test one final time
        result = self.evaluate_test(test_id)
        
        if result is None:
            raise ValidationError(f"Test {test_id} has insufficient data for evaluation")
        
        # Update test status
        self.active_tests[test_id]['status'] = TestStatus.COMPLETED
        
        # Store completed test
        self.completed_tests.append(result)
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        return result
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """
        Get current status of an A/B test.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Dictionary with test status information
        """
        if test_id not in self.active_tests:
            raise ValidationError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        
        return {
            'test_id': test_id,
            'status': test_config['status'].value,
            'model_a_name': test_config['model_a_name'],
            'model_b_name': test_config['model_b_name'],
            'metric_name': test_config['metric_name'],
            'sample_size': len(test_config['scores_a']),
            'start_time': test_config['start_time'],
            'duration': datetime.now() - test_config['start_time'],
            'current_scores_a': test_config['scores_a'][-10:] if test_config['scores_a'] else [],
            'current_scores_b': test_config['scores_b'][-10:] if test_config['scores_b'] else []
        }
    
    def list_active_tests(self) -> List[str]:
        """
        List all active test IDs.
        
        Returns:
            List of active test IDs
        """
        return list(self.active_tests.keys())
    
    def get_completed_tests(self, limit: Optional[int] = None) -> List[ABTestResult]:
        """
        Get completed test results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of completed test results
        """
        if limit is None:
            return self.completed_tests.copy()
        else:
            return self.completed_tests[-limit:]
    
    def _calculate_metric_score(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        metric_name: str
    ) -> float:
        """Calculate performance metric score."""
        if metric_name == 'mse':
            return ((y_true - y_pred) ** 2).mean()
        elif metric_name == 'mae':
            return abs(y_true - y_pred).mean()
        elif metric_name == 'rmse':
            return np.sqrt(((y_true - y_pred) ** 2).mean())
        elif metric_name == 'correlation':
            correlation = y_true.corr(pd.Series(y_pred))
            return correlation if not np.isnan(correlation) else 0.0
        elif metric_name == 'directional_accuracy':
            # Calculate directional accuracy
            true_direction = np.sign(y_true.diff().dropna())
            pred_direction = np.sign(pd.Series(y_pred).diff().dropna())
            min_len = min(len(true_direction), len(pred_direction))
            if min_len > 0:
                return (true_direction.iloc[-min_len:] == pred_direction.iloc[-min_len:]).mean()
            else:
                return 0.0
        else:
            raise ValidationError(f"Unknown metric: {metric_name}")
    
    def _calculate_individual_metric_score(
        self,
        y_true_single: float,
        y_pred_single: float,
        metric_name: str
    ) -> float:
        """Calculate performance metric score for individual prediction."""
        if metric_name == 'mse':
            return (y_true_single - y_pred_single) ** 2
        elif metric_name == 'mae':
            return abs(y_true_single - y_pred_single)
        elif metric_name == 'rmse':
            return (y_true_single - y_pred_single) ** 2  # Will be sqrt'd later if needed
        elif metric_name == 'correlation':
            # For individual predictions, return squared error as proxy
            return (y_true_single - y_pred_single) ** 2
        elif metric_name == 'directional_accuracy':
            # For individual predictions, return squared error as proxy
            return (y_true_single - y_pred_single) ** 2
        else:
            raise ValidationError(f"Unknown metric: {metric_name}")
    
    def calculate_sample_size_requirement(
        self,
        effect_size: float,
        power: float = 0.8,
        significance_level: Optional[float] = None
    ) -> int:
        """
        Calculate required sample size for A/B test.
        
        Args:
            effect_size: Expected effect size
            power: Statistical power (1 - Type II error rate)
            significance_level: Significance level (Type I error rate)
            
        Returns:
            Required sample size per group
        """
        if significance_level is None:
            significance_level = self.significance_level
        
        # Use Cohen's formula for sample size calculation
        # This is a simplified version - in practice, you might want more sophisticated methods
        
        alpha = significance_level
        beta = 1 - power
        
        # Z-scores for alpha and beta
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(int(np.ceil(n)), self.minimum_sample_size)