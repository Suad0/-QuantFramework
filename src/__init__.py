"""
Professional Quantitative Framework

A professional-grade, production-ready quantitative finance platform with advanced
data management, sophisticated modeling capabilities, robust risk management,
comprehensive backtesting, and enterprise-level architecture patterns.
"""

__version__ = "1.0.0"

# Convenience imports for common components
from .domain.entities import Portfolio, Position, Strategy
from .domain.value_objects import Price, Return, Signal, RiskMetrics, PerformanceMetrics
from .domain.exceptions import QuantFrameworkError
from .infrastructure.container import get_container
from .infrastructure.container_config import initialize_application