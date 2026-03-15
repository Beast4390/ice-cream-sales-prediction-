"""Train a regression model to predict ice cream sales/revenue."""

from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_clean_data, prepare_features_targets
from utils import summarize_data


def create_output_dirs(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def create_plots(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: str,
    feature_name: str,
    target_name: str,
) -> None:
    """Create and save exploratory plots."""

    scatter_path = os.path.join(output_dir, "scatter_plot.png")
    reg_path = os.path.join(output_dir, "regression_plot.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test.iloc[:, 0], y_test, color="navy", alpha=0.7, label="Actual")
    plt.scatter(X_test.iloc[:, 0], y_pred, color="orange", alpha=0.7, label="Predicted")
    plt.title("Actual vs Predicted")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test.iloc[:, 0], y_test, color="tab:blue", alpha=0.6, label="Actual")
    plt.plot(
        X_test.iloc[:, 0], y_pred, color="tab:orange", linewidth=2, label="Predicted trend"
    )
    plt.title("Regression Line")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(reg_path)
    plt.close()


def train_from_df(
    df: pd.DataFrame,
    model_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[LinearRegression, dict]:
    """Train a regression model from a DataFrame and save artifacts.

    Returns the trained model and a dictionary of performance metrics.
    """

    create_output_dirs(output_dir)

    df = df.dropna().reset_index(drop=True)

    # Show dataset summary for transparency.
    summary = summarize_data(df)
    print("Dataset shape:", summary["shape"])
    print("Missing values:\n", summary["missing_values"])

    X, y, meta = prepare_features_targets(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"\nModel evaluation")
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    create_plots(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        output_dir=output_dir,
        feature_name=X.columns[0],
        target_name=meta["target_column"],
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nSaved trained model to: {model_path}")
    print(f"Saved plots to: {output_dir}")

    metrics = {
        "r2": r2,
        "mse": mse,
        "n_samples": len(df),
        "feature": meta.get("feature_columns"),
        "target": meta.get("target_column"),
    }

    return model, metrics


def train(csv_path: str, model_path: str, output_dir: str) -> None:
    """Train a regression model and save artifacts."""

    df = load_and_clean_data(csv_path)
    train_from_df(df, model_path=model_path, output_dir=output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ice cream sales regression model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/IceCreamData.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/icecream_model.pkl",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="outputs",
        help="Directory to save output plots.",
    )

    args = parser.parse_args()

    train(csv_path=args.data, model_path=args.model, output_dir=args.outputs)
