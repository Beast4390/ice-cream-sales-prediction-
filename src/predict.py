"""Reusable prediction script for Ice Cream Sales Prediction."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd


def load_model(model_path: str):
    """Load a trained model from disk."""

    model = joblib.load(model_path)
    return model


def predict_from_values(model, feature_df: pd.DataFrame) -> pd.Series:
    """Predict target values for the provided feature DataFrame."""

    return pd.Series(model.predict(feature_df), index=feature_df.index)


def build_dataframe(temperature: float) -> pd.DataFrame:
    """Create a single-row DataFrame with the required feature(s)."""

    return pd.DataFrame({"Temperature": [temperature]})


def main() -> None:
    """Command-line entry point for predictions."""

    parser = argparse.ArgumentParser(
        description="Predict Ice Cream Sales/Revenue using a saved model."
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature value for which to predict ice cream revenue.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/icecream_model.pkl",
        help="Path to the saved model file.",
    )

    args = parser.parse_args()

    model = load_model(args.model)
    input_df = build_dataframe(args.temperature)
    prediction = predict_from_values(model, input_df)

    print(f"Input temperature: {args.temperature}")
    print(f"Predicted revenue: {prediction.iloc[0]:.2f}")


if __name__ == "__main__":
    main()
