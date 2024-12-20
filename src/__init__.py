"""
AI Budget Forecasting System
===========================

A system for forecasting store profits using various machine learning models.

This package contains the following modules:
- data_preparation: Functions for loading and preprocessing data
- model_selection: Functions for selecting the best forecasting model
- outlier_detection: Functions for detecting and handling outliers
- prediction: Functions for making future predictions
- utils: Utility functions used across modules
"""

__version__ = '1.0.0'
__author__ = 'Varees Adulyasas'
__email__ = 'vareesadulyasas@gmail.com'

from .data_preparation import DataPreparation
from .model_selection import ModelSelection
from .outlier_detection import OutlierDetector
from .prediction import Predictor
from .utils import create_dataset, evaluate_predictions

__all__ = [
    'DataPreparation',
    'ModelSelection',
    'OutlierDetector',
    'Predictor',
    'create_dataset',
    'evaluate_predictions'
]