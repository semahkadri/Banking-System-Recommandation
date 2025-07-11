"""
Data utility functions for the Bank Check Prediction System

This module provides common data processing functions used throughout
the application for loading, cleaning, and transforming data.
"""

import csv
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path


def safe_read_csv(
    filename: str, 
    encoding: str = 'utf-8',
    fallback_encoding: str = 'latin-1'
) -> List[Dict[str, Any]]:
    """
    Safely read CSV file with error handling and encoding detection.
    
    Args:
        filename: Path to CSV file
        encoding: Primary encoding to try
        fallback_encoding: Fallback encoding if primary fails
        
    Returns:
        List of dictionaries representing CSV rows
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    print(f"Reading CSV file: {filename}")
    
    try:
        with open(filename, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        print(f"Successfully read {len(data)} rows with {encoding} encoding")
        return data
    except UnicodeDecodeError:
        print(f"Failed to read with {encoding}, trying {fallback_encoding}")
        try:
            with open(filename, 'r', encoding=fallback_encoding) as f:
                reader = csv.DictReader(f)
                data = list(reader)
            print(f"Successfully read {len(data)} rows with {fallback_encoding} encoding")
            return data
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return []
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []


def safe_read_excel(
    filename: str,
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Safely read Excel file with error handling.
    
    Args:
        filename: Path to Excel file
        sheet_name: Optional sheet name to read
        
    Returns:
        pandas DataFrame
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    print(f"Reading Excel file: {filename}")
    
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        print(f"Successfully read Excel file with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading Excel file {filename}: {e}")
        return pd.DataFrame()


def normalize_feature(values: List[float]) -> Tuple[List[float], float, float]:
    """
    Normalize feature values to [0, 1] range using min-max scaling.
    
    Args:
        values: List of numerical values to normalize
        
    Returns:
        Tuple of (normalized_values, min_value, max_value)
    """
    if not values:
        return values, 0, 1
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        print("All values are identical, returning zeros")
        return [0] * len(values), min_val, max_val
    
    normalized = [(val - min_val) / (max_val - min_val) for val in values]
    print(f"Normalized {len(values)} values from range [{min_val:.2f}, {max_val:.2f}] to [0, 1]")
    
    return normalized, min_val, max_val


def encode_categorical(data: List[Dict[str, Any]], column: str) -> Dict[str, int]:
    """
    Encode categorical variables to numerical values.
    
    Args:
        data: List of dictionaries containing the data
        column: Column name to encode
        
    Returns:
        Dictionary mapping categorical values to numerical codes
    """
    unique_values = list(set(row.get(column, '') for row in data))
    unique_values = [val for val in unique_values if val]  # Remove empty strings
    
    encoding = {val: i for i, val in enumerate(sorted(unique_values))}
    
    print(f"Encoded column '{column}' with {len(encoding)} unique values")
    
    return encoding


def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """
    Calculate regression metrics for model evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MSE, MAE, and R² metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    n = len(y_true)
    
    # Mean Squared Error
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    
    # Mean Absolute Error
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    
    # R-squared
    y_mean = sum(y_true) / n
    ss_tot = sum((y_true[i] - y_mean) ** 2 for i in range(n))
    ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Root Mean Squared Error
    rmse = mse ** 0.5
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def validate_data_completeness(
    data: List[Dict[str, Any]], 
    required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that data contains all required columns.
    
    Args:
        data: List of dictionaries representing data rows
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    if not data:
        return False, required_columns
    
    available_columns = set(data[0].keys())
    missing_columns = [col for col in required_columns if col not in available_columns]
    
    is_valid = len(missing_columns) == 0
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
    else:
        print("All required columns present")
    
    return is_valid, missing_columns


def format_currency_tnd(amount: float, precision: int = 2) -> str:
    """
    Format amount in Tunisian Dinar (TND) currency.
    
    Args:
        amount: Amount to format
        precision: Number of decimal places (default: 2)
        
    Returns:
        Formatted string with TND currency symbol
    """
    try:
        if amount >= 1000000:
            # For millions, show in M TND
            return f"{amount/1000000:,.{max(1, precision-1)}f}M TND"
        elif amount >= 1000:
            # For thousands, show in K TND  
            return f"{amount/1000:,.{max(1, precision-1)}f}K TND"
        else:
            # Regular format
            return f"{amount:,.{precision}f} TND"
    except (ValueError, TypeError):
        return "0.00 TND"


def clean_numeric_data(value: Any, default: float = 0.0) -> float:
    """
    Clean and convert data to numeric format.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value
    """
    if value is None or value == '':
        return default
    
    try:
        # Handle string representations
        if isinstance(value, str):
            # Remove common formatting
            value = value.replace(',', '').replace(' ', '').replace('%', '')
        
        return float(value)
    except (ValueError, TypeError):
        print(f"Could not convert '{value}' to float, using default {default}")
        return default