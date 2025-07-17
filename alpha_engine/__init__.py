"""
Multi-Factor Equity Alpha Engine

A production-quality Python framework for quantitative equity research,
featuring multi-factor modeling, machine learning, and portfolio optimization.

Author: AI Assistant
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Alpha Engine Team"

from .data import DataLoader, DataProcessor
from .features import FeatureEngine
from .models import RidgeModel, XGBoostModel, NeuralNetworkModel
from .portfolio import PortfolioOptimizer, KellySizer
from .backtest import Backtester, PerformanceAnalyzer

__all__ = [
    "DataLoader",
    "DataProcessor", 
    "FeatureEngine",
    "RidgeModel",
    "XGBoostModel", 
    "NeuralNetworkModel",
    "PortfolioOptimizer",
    "KellySizer",
    "Backtester",
    "PerformanceAnalyzer"
]
