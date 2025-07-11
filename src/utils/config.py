"""
Simple configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load basic configuration."""
    return {
        "data": {
            "raw_data_path": "data/raw",
            "processed_data_path": "data/processed", 
            "models_path": "data/models",
        },
        "model": {
            "learning_rate": 0.01,
            "epochs": 1000,
        },
        "logging": {
            "level": "INFO",
        }
    }

def get_data_paths() -> Dict[str, Path]:
    """Get standardized data paths."""
    base_path = Path.cwd()
    
    return {
        "raw": base_path / "data/raw",
        "processed": base_path / "data/processed",
        "models": base_path / "data/models",
    }