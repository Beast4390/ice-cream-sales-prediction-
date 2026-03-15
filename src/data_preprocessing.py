"""Data loading and preprocessing for Ice Cream Sales Prediction."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from utils import detect_target_column, load_data


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset and perform basic cleaning.

    Steps:
      - Load CSV from disk.
      - Drop rows with missing values.
      - Reset index.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        Cleaned pandas DataFrame.
    """

    df = load_data(csv_path)

    # Drop any rows with missing values; this dataset is small and this keeps things simple.
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_features_targets(
    df: pd.DataFrame, target_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Prepare feature matrix X and target vector y.

    Args:
        df: Cleaned dataframe.
        target_column: Optional column name to use as the target. If None, the target will be detected.

    Returns:
        X: DataFrame of features.
        y: Series of target values.
        meta: A small dictionary describing the feature and target columns.
    """

    if target_column is None:
        target_column = detect_target_column(df)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    meta = {"target_column": target_column, "feature_columns": ", ".join(X.columns)}
    return X, y, meta
