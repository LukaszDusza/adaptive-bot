# validators.py
"""
Common validation functions for defensive programming.

Provides reusable validators for:
- DataFrame validation (non-empty, required columns, no NaN/inf)
- Numeric value validation (positive, in range)
- API response validation
- Model output validation
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Union, Any, Dict

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_dataframe_not_empty(
    df: pd.DataFrame,
    name: str = "DataFrame",
    min_rows: int = 1
) -> None:
    """
    Validate that DataFrame is not empty.

    Args:
        df: DataFrame to validate
        name: Name for error messages
        min_rows: Minimum number of rows required

    Raises:
        ValidationError: If DataFrame is empty or has too few rows
    """
    if df is None:
        raise ValidationError(f"{name} is None")

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} is not a DataFrame (got {type(df).__name__})")

    if len(df) == 0:
        raise ValidationError(f"{name} is empty (0 rows)")

    if len(df) < min_rows:
        raise ValidationError(f"{name} has only {len(df)} rows (need at least {min_rows})")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame has all required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages

    Raises:
        ValidationError: If any required columns are missing
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{name} missing required columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def validate_no_nan_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    name: str = "DataFrame",
    max_nan_pct: float = 0.0
) -> None:
    """
    Validate that DataFrame has no (or few) NaN values.

    Args:
        df: DataFrame to validate
        columns: Columns to check (None = all columns)
        name: Name for error messages
        max_nan_pct: Maximum percentage of NaN values allowed (0.0 = no NaN)

    Raises:
        ValidationError: If NaN values exceed threshold
    """
    cols_to_check = columns if columns is not None else df.columns

    for col in cols_to_check:
        if col not in df.columns:
            continue

        nan_count = df[col].isna().sum()
        nan_pct = nan_count / len(df) if len(df) > 0 else 0.0

        if nan_pct > max_nan_pct:
            raise ValidationError(
                f"{name}['{col}'] has {nan_count} NaN values "
                f"({nan_pct:.1%} > {max_nan_pct:.1%} threshold)"
            )


def validate_no_infinite_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame has no infinite values.

    Args:
        df: DataFrame to validate
        columns: Columns to check (None = all numeric columns)
        name: Name for error messages

    Raises:
        ValidationError: If infinite values are found
    """
    if columns is None:
        # Check all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = columns

    for col in numeric_cols:
        if col not in df.columns:
            continue

        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            raise ValidationError(
                f"{name}['{col}'] has {inf_count} infinite values"
            )


def validate_positive(
    value: Union[int, float],
    name: str = "Value",
    allow_zero: bool = False
) -> None:
    """
    Validate that a numeric value is positive.

    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether zero is allowed

    Raises:
        ValidationError: If value is not positive
    """
    if value is None:
        raise ValidationError(f"{name} is None")

    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} is not numeric (got {type(value).__name__})")

    if np.isnan(value):
        raise ValidationError(f"{name} is NaN")

    if np.isinf(value):
        raise ValidationError(f"{name} is infinite")

    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be >= 0 (got {value})")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be > 0 (got {value})")


def validate_in_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "Value"
) -> None:
    """
    Validate that a numeric value is in a specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)
        name: Name for error messages

    Raises:
        ValidationError: If value is out of range
    """
    if value is None:
        raise ValidationError(f"{name} is None")

    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} is not numeric (got {type(value).__name__})")

    if np.isnan(value):
        raise ValidationError(f"{name} is NaN")

    if np.isinf(value):
        raise ValidationError(f"{name} is infinite")

    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val} (got {value})")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val} (got {value})")


def validate_probability(
    value: float,
    name: str = "Probability"
) -> None:
    """
    Validate that a value is a valid probability (0.0 to 1.0).

    Args:
        value: Probability to validate
        name: Name for error messages

    Raises:
        ValidationError: If value is not a valid probability
    """
    validate_in_range(value, min_val=0.0, max_val=1.0, name=name)


def validate_api_response(
    response: Any,
    expected_type: type = dict,
    required_keys: Optional[List[str]] = None,
    name: str = "API response"
) -> None:
    """
    Validate API response structure.

    Args:
        response: API response to validate
        expected_type: Expected response type (dict, list, etc.)
        required_keys: Required keys if response is dict
        name: Name for error messages

    Raises:
        ValidationError: If response is invalid
    """
    if response is None:
        raise ValidationError(f"{name} is None")

    if not isinstance(response, expected_type):
        raise ValidationError(
            f"{name} has wrong type: expected {expected_type.__name__}, "
            f"got {type(response).__name__}"
        )

    if expected_type == dict and required_keys:
        missing = set(required_keys) - set(response.keys())
        if missing:
            raise ValidationError(
                f"{name} missing required keys: {sorted(missing)}. "
                f"Available: {sorted(response.keys())}"
            )


def validate_model_predictions(
    predictions: np.ndarray,
    expected_shape: Optional[tuple] = None,
    is_probability: bool = True,
    name: str = "Model predictions"
) -> None:
    """
    Validate model prediction output.

    Args:
        predictions: Model predictions array
        expected_shape: Expected shape (None = don't check)
        is_probability: Whether predictions should be in [0, 1] range
        name: Name for error messages

    Raises:
        ValidationError: If predictions are invalid
    """
    if predictions is None:
        raise ValidationError(f"{name} is None")

    if not isinstance(predictions, np.ndarray):
        raise ValidationError(
            f"{name} is not numpy array (got {type(predictions).__name__})"
        )

    if len(predictions) == 0:
        raise ValidationError(f"{name} is empty")

    if expected_shape is not None and predictions.shape != expected_shape:
        raise ValidationError(
            f"{name} has wrong shape: expected {expected_shape}, "
            f"got {predictions.shape}"
        )

    if np.isnan(predictions).any():
        nan_count = np.isnan(predictions).sum()
        raise ValidationError(f"{name} has {nan_count} NaN values")

    if np.isinf(predictions).any():
        inf_count = np.isinf(predictions).sum()
        raise ValidationError(f"{name} has {inf_count} infinite values")

    if is_probability:
        if (predictions < 0).any() or (predictions > 1).any():
            raise ValidationError(
                f"{name} contains values outside [0, 1] range "
                f"(min={predictions.min():.3f}, max={predictions.max():.3f})"
            )


# Convenience function for common OHLCV validation
def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    min_rows: int = 100,
    max_nan_pct: float = 0.1,
    name: str = "OHLCV DataFrame"
) -> None:
    """
    Validate OHLCV DataFrame for trading.

    Checks:
    - Not empty (>= min_rows)
    - Has OHLC columns
    - No infinite values
    - Limited NaN values
    - Valid OHLC relationships

    Args:
        df: OHLCV DataFrame to validate
        min_rows: Minimum number of rows required
        max_nan_pct: Maximum percentage of NaN values per column
        name: Name for error messages

    Raises:
        ValidationError: If DataFrame is invalid
    """
    # Check not empty
    validate_dataframe_not_empty(df, name=name, min_rows=min_rows)

    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    validate_required_columns(df, required_cols, name=name)

    # Check no infinite values
    validate_no_infinite_values(df, columns=required_cols, name=name)

    # Check limited NaN values
    validate_no_nan_values(df, columns=required_cols, name=name, max_nan_pct=max_nan_pct)

    # Check OHLC relationships
    invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
    if invalid_high > 0:
        raise ValidationError(
            f"{name} has {invalid_high} rows where high < max(open, close)"
        )

    invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
    if invalid_low > 0:
        raise ValidationError(
            f"{name} has {invalid_low} rows where low > min(open, close)"
        )

    # Check volume is positive
    non_positive_volume = (df['volume'] <= 0).sum()
    if non_positive_volume > 0:
        raise ValidationError(
            f"{name} has {non_positive_volume} rows with non-positive volume"
        )
