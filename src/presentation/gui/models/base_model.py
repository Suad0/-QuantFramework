"""
Base model classes for the MVC pattern in the PyQt application.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd


class BaseModel(QObject):
    """Base model class for MVC pattern."""
    
    # Signals for model changes
    data_changed = pyqtSignal()
    error_occurred = pyqtSignal(str)
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._data = {}
        self._is_loading = False
    
    @property
    def is_loading(self) -> bool:
        """Check if model is currently loading data."""
        return self._is_loading
    
    def set_loading(self, loading: bool):
        """Set loading state."""
        if self._is_loading != loading:
            self._is_loading = loading
            if loading:
                self.loading_started.emit()
            else:
                self.loading_finished.emit()
    
    def get_data(self, key: str = None) -> Any:
        """Get data from the model."""
        if key is None:
            return self._data
        return self._data.get(key)
    
    def set_data(self, key: str, value: Any):
        """Set data in the model."""
        self._data[key] = value
        self.data_changed.emit()
    
    def clear_data(self):
        """Clear all data from the model."""
        self._data.clear()
        self.data_changed.emit()
    
    def emit_error(self, error_message: str):
        """Emit error signal."""
        self.error_occurred.emit(error_message)


class AnalysisModel(BaseModel):
    """Model for quantitative analysis data."""
    
    # Additional signals specific to analysis
    analysis_completed = pyqtSignal(dict)
    progress_updated = pyqtSignal(str, int)  # message, percentage
    
    def __init__(self):
        super().__init__()
        self._analysis_results = {}
        self._current_parameters = {}
    
    @property
    def analysis_results(self) -> Dict[str, Any]:
        """Get current analysis results."""
        return self._analysis_results
    
    @property
    def current_parameters(self) -> Dict[str, Any]:
        """Get current analysis parameters."""
        return self._current_parameters
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set analysis parameters."""
        self._current_parameters = parameters.copy()
        self.data_changed.emit()
    
    def set_analysis_results(self, results: Dict[str, Any]):
        """Set analysis results."""
        self._analysis_results = results.copy()
        self.analysis_completed.emit(results)
        self.data_changed.emit()
    
    def get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data from results."""
        return self._analysis_results.get('market_data')
    
    def get_signals(self) -> Optional[pd.DataFrame]:
        """Get trading signals from results."""
        return self._analysis_results.get('signals')
    
    def get_portfolio_returns(self) -> Optional[pd.Series]:
        """Get portfolio returns from results."""
        return self._analysis_results.get('portfolio_returns')
    
    def get_risk_metrics(self) -> Optional[Dict[str, float]]:
        """Get risk metrics from results."""
        return self._analysis_results.get('risk_metrics')
    
    def get_portfolio_weights(self) -> Optional[Dict[str, float]]:
        """Get portfolio weights from results."""
        return self._analysis_results.get('portfolio_weights')
    
    def update_progress(self, message: str, percentage: int):
        """Update analysis progress."""
        self.progress_updated.emit(message, percentage)
    
    def clear_results(self):
        """Clear analysis results."""
        self._analysis_results.clear()
        self.data_changed.emit()


class PortfolioModel(BaseModel):
    """Model for portfolio data."""
    
    portfolio_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self._portfolios = {}
        self._current_portfolio_id = None
    
    def add_portfolio(self, portfolio_id: str, portfolio_data: Dict[str, Any]):
        """Add a portfolio to the model."""
        self._portfolios[portfolio_id] = portfolio_data
        self.portfolio_updated.emit(portfolio_data)
        self.data_changed.emit()
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio by ID."""
        return self._portfolios.get(portfolio_id)
    
    def get_all_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """Get all portfolios."""
        return self._portfolios.copy()
    
    def set_current_portfolio(self, portfolio_id: str):
        """Set the current active portfolio."""
        if portfolio_id in self._portfolios:
            self._current_portfolio_id = portfolio_id
            self.data_changed.emit()
    
    def get_current_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get the current active portfolio."""
        if self._current_portfolio_id:
            return self._portfolios.get(self._current_portfolio_id)
        return None
    
    def remove_portfolio(self, portfolio_id: str):
        """Remove a portfolio."""
        if portfolio_id in self._portfolios:
            del self._portfolios[portfolio_id]
            if self._current_portfolio_id == portfolio_id:
                self._current_portfolio_id = None
            self.data_changed.emit()


class SettingsModel(BaseModel):
    """Model for application settings."""
    
    settings_changed = pyqtSignal(str, object)  # setting_name, value
    
    def __init__(self):
        super().__init__()
        self._settings = {
            'theme': 'light',
            'auto_refresh': True,
            'refresh_interval': 60,
            'default_strategy': 'momentum',
            'default_optimization': 'mean_variance',
            'risk_tolerance': 'moderate',
            'chart_style': 'candlestick',
            'export_format': 'csv'
        }
    
    def get_setting(self, setting_name: str, default_value: Any = None) -> Any:
        """Get a setting value."""
        return self._settings.get(setting_name, default_value)
    
    def set_setting(self, setting_name: str, value: Any):
        """Set a setting value."""
        old_value = self._settings.get(setting_name)
        if old_value != value:
            self._settings[setting_name] = value
            self.settings_changed.emit(setting_name, value)
            self.data_changed.emit()
    
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings."""
        return self._settings.copy()
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update multiple settings."""
        for key, value in settings.items():
            self.set_setting(key, value)
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        default_settings = {
            'theme': 'light',
            'auto_refresh': True,
            'refresh_interval': 60,
            'default_strategy': 'momentum',
            'default_optimization': 'mean_variance',
            'risk_tolerance': 'moderate',
            'chart_style': 'candlestick',
            'export_format': 'csv'
        }
        self._settings = default_settings.copy()
        self.data_changed.emit()