"""
Bank Check Prediction System

A machine learning system for predicting bank check issuance
and maximum authorized amounts per client.

This package provides:
- Data processing and analysis tools
- Machine learning models for prediction
- Streamlit dashboard for visualization

Version: 1.0.0
"""

__version__ = "1.0.0"

# Package imports
from .models import CheckPredictionModel

__all__ = [
    "CheckPredictionModel",
]