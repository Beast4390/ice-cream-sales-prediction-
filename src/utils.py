"""Utility helpers for ice cream sales prediction.

This module provides data loading, summarization, and helper functions to keep the
project modular and beginner-friendly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Args:
        csv_path: Path to the CSV dataset.

    Returns:
        A pandas DataFrame loaded from the CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def summarize_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a summary of the dataset.

    Returns a dictionary with common exploratory outputs.
    """
    summary: Dict[str, Any] = {}
    summary["head"] = df.head().copy()
    summary["shape"] = df.shape
    summary["info"] = df.info(buf=None)
    summary["describe"] = df.describe(include="all")
    summary["missing_values"] = df.isna().sum()

    return summary


def detect_target_column(df: pd.DataFrame, preferred_targets: Optional[List[str]] = None) -> str:
    """Detect the most likely target column for regression.

    Args:
        df: DataFrame containing the dataset.
        preferred_targets: Optional list of column names to prefer if present.

    Returns:
        Name of the detected target column.

    Raises:
        ValueError: If no suitable target column can be detected.
    """

    if preferred_targets is None:
        preferred_targets = ["revenue", "sales", "income", "profit"]

    # Case-insensitive search for preferred target names.
    lower_cols = {c.lower(): c for c in df.columns}
    for name in preferred_targets:
        if name in lower_cols:
            return lower_cols[name]

    # Fall back to numeric columns.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found to use as a target.")

    # If there are multiple numeric columns, pick the one with the highest variance (common heuristic).
    if len(numeric_cols) > 1:
        variances = df[numeric_cols].var(dropna=True)
        target_col = variances.idxmax()
    else:
        target_col = numeric_cols[0]

    return target_col


def validate_numeric_input(value: Any, name: str) -> float:
    """Validate a numeric input value from the user or CLI.

    Args:
        value: The raw input value.
        name: Name of the input (for error messages).

    Returns:
        The value converted to float.

    Raises:
        ValueError: If the input cannot be converted to float.
    """

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {name}: {value}") from exc
