"""Professional Streamlit app for Ice Cream Sales Prediction."""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score

# Ensure `src/` is importable when running via Streamlit.
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from train_model import train_from_df


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "IceCreamData.csv"
MODEL_PATH = BASE_DIR / "models" / "icecream_model.pkl"

# Exchange rate: 1 USD = 92.513 INR (as of March 15, 2026)
USD_TO_INR = 92.513

# Average price per ice cream unit (in USD)
AVG_PRICE_PER_UNIT = 5.0


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path, model_mtime: float):
    """Load the trained model, with cache invalidation when the file changes."""

    return joblib.load(model_path)


def create_html_report(
    df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    model_metrics: dict | None,
    peak_temp: float,
    peak_revenue_inr: float,
    figures: dict[str, go.Figure],
) -> bytes:
    """Build an HTML sales forecast report for download."""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    r2 = model_metrics.get("r2") if model_metrics else None
    mse = model_metrics.get("mse") if model_metrics else None
    n_samples = model_metrics.get("n_samples") if model_metrics else len(df)

    # Build a lightweight report using Plotly HTML snippets.
    report_sections = []

    report_sections.append(f"<h1>Ice Cream Sales Forecast Report</h1>")
    report_sections.append(f"<p>Generated: {ts}</p>")

    report_sections.append("<h2>Key Metrics</h2>")
    report_sections.append(
        "<ul>"
        f"<li>Total records: {len(df)}</li>"
        f"<li>Peak sales temperature: {peak_temp:.1f}°C</li>"
        f"<li>Peak revenue: ₹{peak_revenue_inr:,.0f}</li>"
        f"<li>R² score: {r2:.3f}</li>" if r2 is not None else ""
        f"<li>Mean squared error: {mse:.2f}</li>" if mse is not None else ""
        f"<li>Training samples: {n_samples}</li>"
        "</ul>"
    )

    report_sections.append("<h2>Scenario Simulations</h2>")
    report_sections.append(scenario_df.to_html(index=False, classes="table table-striped"))

    report_sections.append("<h2>Charts</h2>")

    # Embed charts; include plotly.js once.
    report_html = "".join(report_sections)
    report_html += pio.to_html(figures["forecast"], include_plotlyjs="cdn", full_html=False)

    # Append remaining figures without plotly.js
    for key in ["scatter", "regression", "residual", "hist", "trend", "pie", "segment"]:
        if key in figures:
            report_html += pio.to_html(figures[key], include_plotlyjs=False, full_html=False)

    full_html = f"<html><head><meta charset=\"utf-8\"><title>Ice Cream Sales Forecast Report</title></head><body>{report_html}</body></html>"

    return full_html.encode("utf-8")


def main() -> None:
    st.set_page_config(
        page_title="Ice Cream Sales Prediction",
        page_icon="🍦",
        layout="wide",
    )

    st.markdown(
        """
        <h1 style='text-align: center;'>🍦 Ice Cream Sales Prediction Dashboard</h1>
        <p style='text-align: center; font-size: 18px; color: #9aa0a6;'>
            Predict ice cream revenue from temperature and explore the dataset interactively
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    show_data = st.sidebar.checkbox("Show raw dataset")
    show_stats = st.sidebar.checkbox("Show summary statistics", value=True)

    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset (CSV)",
        type=["csv"],
        help="Upload a CSV with at least 'Temperature' and 'Revenue' columns to override the default dataset.",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Custom dataset loaded successfully.")
        except Exception as exc:
            st.sidebar.error(f"Failed to load uploaded file: {exc}")
            return
    else:
        try:
            df = load_dataset(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Dataset not found: {DATA_PATH}")
            return

    if not MODEL_PATH.exists():
        # Provide a friendly call-to-action for first-time usage.
        st.warning("Model not found. A model will be trained automatically when you upload a dataset.")

    model = None
    model_metrics = None

    def retrain_and_load_model(dataset: pd.DataFrame) -> tuple:
        """Retrain model on the provided dataset and return (model, metrics)."""

        # Ensure we keep the dataset clean and focused on required columns.
        dataset = dataset.copy()
        if "Temperature" not in dataset.columns or "Revenue" not in dataset.columns:
            raise ValueError("Dataset must contain 'Temperature' and 'Revenue' columns.")

        # Train model and persist it for later runs.
        trained_model, metrics = train_from_df(
            dataset,
            model_path=str(MODEL_PATH),
            output_dir=str(BASE_DIR / "outputs"),
        )

        return trained_model, metrics

    if uploaded_file is not None:
        # When the user uploads a file, retrain and refresh insights.
        try:
            df = pd.read_csv(uploaded_file)
            model, model_metrics = retrain_and_load_model(df)
            st.sidebar.success("Custom dataset loaded and model retrained.")
        except Exception as exc:
            st.sidebar.error(f"Failed to load or train on uploaded file: {exc}")
            return
    else:
        try:
            df = load_dataset(DATA_PATH)
        except FileNotFoundError:
            st.error(f"Dataset not found: {DATA_PATH}")
            return

        if MODEL_PATH.exists():
            model = load_model(MODEL_PATH, MODEL_PATH.stat().st_mtime)
        else:
            # If no saved model exists, train one on the default dataset.
            model, model_metrics = retrain_and_load_model(df)

    if "Temperature" not in df.columns or "Revenue" not in df.columns:
        st.error("Dataset must contain 'Temperature' and 'Revenue' columns.")
        return

    # Model performance & peak insights
    peak_row = df.loc[df["Revenue"].idxmax()]
    peak_temp = peak_row["Temperature"]
    peak_revenue_inr = peak_row["Revenue"] * USD_TO_INR

    # If metrics were computed during training, reuse them; otherwise compute here.
    if model_metrics is None:
        y_true = df["Revenue"]
        y_pred = model.predict(df[["Temperature"]])
        model_r2 = r2_score(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_samples = len(df)
    else:
        model_r2 = model_metrics.get("r2", 0.0)
        model_mse = model_metrics.get("mse", 0.0)
        model_samples = model_metrics.get("n_samples", len(df))

    st.markdown("---")

    # Top metric cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Records", len(df))
    with m2:
        st.metric("Average Temperature", f"{df['Temperature'].mean():.2f} °C")
    with m3:
        st.metric("Average Revenue", f"₹{df['Revenue'].mean() * USD_TO_INR:.2f}")
    with m4:
        st.metric("Average Sales Count", f"{(df['Revenue'] / AVG_PRICE_PER_UNIT).mean():.0f} units")
    with m5:
        total_sales_count = (df['Revenue'] / AVG_PRICE_PER_UNIT).sum()
        st.metric("Total Sales Count", f"{total_sales_count:.0f} units")
    with m6:
        st.metric(
            "Model Accuracy (R²)",
            f"{model_r2:.2f}",
            help="R² score computed on the available dataset",
        )

    with st.expander("Model Performance Details"):
        st.write(f"**Mean Squared Error:** {model_mse:.2f}")
        st.write(f"**Training samples used:** {model_samples}")

    st.markdown(f"> **Peak revenue observed near {peak_temp:.1f}°C (₹{peak_revenue_inr:,.0f}).**")

    st.markdown("---")

    # Enhanced Business Prediction Section
    st.subheader("💰 Revenue Forecast & Business Insights")

    # Business Input Controls
    st.markdown("### 📊 Business Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.slider(
            "Temperature (°C)",
            min_value=float(df["Temperature"].min()),
            max_value=float(df["Temperature"].max()),
            value=float(df["Temperature"].mean()),
            step=0.1,
            help="Select the temperature for revenue prediction"
        )

    with col2:
        selling_price = st.slider(
            "Selling Price per Unit (₹)",
            min_value=5.0,
            max_value=1000.0,
            value=50.0,
            step=5.0,
            help="Current selling price per ice cream unit in INR"
        )

    with col3:
        cost_price = st.slider(
            "Cost per Unit (₹)",
            min_value=1.0,
            max_value=500.0,
            value=20.0,
            step=1.0,
            help="Production cost per ice cream unit in INR"
        )

    # Prediction Calculations
    input_df = pd.DataFrame({"Temperature": [temperature]})
    predicted_revenue_usd = model.predict(input_df)[0]
    predicted_revenue_inr = predicted_revenue_usd * USD_TO_INR

    estimated_sales_count = int(predicted_revenue_inr / selling_price)
    estimated_profit = (selling_price - cost_price) * estimated_sales_count

    # Prediction KPI Cards
    st.markdown("### 🎯 Forecast Results")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric(
            "Revenue Forecast",
            f"₹{predicted_revenue_inr:,.0f}",
            help=f"Predicted revenue at {temperature:.1f}°C"
        )

    with kpi2:
        st.metric(
            "Estimated Unit Sales",
            f"{estimated_sales_count:,}",
            help="Units calculated as revenue ÷ selling price"
        )

    with kpi3:
        st.metric(
            "Profit Projection",
            f"₹{estimated_profit:,.0f}",
            help="Net profit after costs"
        )

    st.caption("Note: Unit sales are estimated as `Predicted Revenue ÷ Selling Price`, so changing the selling price will change the implied units sold.")

    # Business Insight Recommendation
    st.markdown("### 💡 Smart Recommendations")

    if temperature > 32:
        demand_level = "High"
        recommendation = (
            "🔥 **High Demand Alert**: Prepare extra stock. "
            "Hot weather usually drives peak ice cream sales."
        )
        bubble_color = "#00ff88"
    elif temperature >= 25:
        demand_level = "Medium"
        recommendation = (
            "📈 **Moderate Demand**: Maintain normal inventory. "
            "Sales are healthy but not at peak levels."
        )
        bubble_color = "#ffd700"
    else:
        demand_level = "Low"
        recommendation = (
            "❄️ **Low Demand Warning**: Reduce stock levels. "
            "Consider promotions to stimulate demand."
        )
        bubble_color = "#ff6b6b"

    st.markdown(
        f"""
        <div style='border-left: 4px solid {bubble_color}; padding: 12px; background-color: rgba(255,255,255,0.04);'
             >
            <strong>{demand_level} Demand Insight</strong><br>
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scenario Analysis Table
    st.markdown("### 📈 Temperature Scenario Analysis")

    def get_demand_level(temp: float) -> str:
        if temp > 32:
            return "High"
        if temp >= 25:
            return "Medium"
        return "Low"

    scenarios = [20, 25, 30, 35, 40]
    scenario_data = []

    for temp in scenarios:
        temp_df = pd.DataFrame({"Temperature": [temp]})
        rev_usd = model.predict(temp_df)[0]
        rev_inr = rev_usd * USD_TO_INR
        sales = int(rev_inr / selling_price)
        profit = (selling_price - cost_price) * sales
        scenario_data.append({
            "Temperature (°C)": temp,
            "Revenue Forecast (₹)": f"{rev_inr:,.0f}",
            "Unit Sales": f"{sales:,}",
            "Profit (₹)": f"{profit:,.0f}",
            "Demand Level": get_demand_level(temp),
        })

    scenario_df = pd.DataFrame(scenario_data)
    st.dataframe(scenario_df, use_container_width=True)

    csv_report = scenario_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast Report",
        data=csv_report,
        file_name="ice_cream_forecast_report.csv",
        mime="text/csv",
        help="Download the scenario analysis data as a CSV file.",
    )

    # Forecast Chart
    st.markdown("### 📊 Revenue Forecast Chart")

    # Generate forecast data across a standard temperature range (0–45°C)
    temp_range = pd.DataFrame({
        "Temperature": np.linspace(0, 45, 100)
    })
    forecast_revenue = model.predict(temp_range) * USD_TO_INR

    fig_forecast = go.Figure()
    fig_forecast.add_trace(
        go.Scatter(
            x=temp_range["Temperature"],
            y=forecast_revenue,
            mode="lines",
            name="Revenue Forecast",
            line=dict(color="#00ff88", width=3)
        )
    )

    # Add current prediction point
    fig_forecast.add_trace(
        go.Scatter(
            x=[temperature],
            y=[predicted_revenue_usd * USD_TO_INR],
            mode="markers",
            name="Current Prediction",
            marker=dict(size=12, color="#ff6b6b", symbol="star")
        )
    )

    fig_forecast.update_layout(
        title="Revenue Forecast by Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="Revenue (₹)",
        template="plotly_dark",
        showlegend=True
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("---")

    # Prepare sorted data for line/regression charts
    sorted_df = df.sort_values("Temperature").copy()
    sorted_df["Predicted_Revenue"] = model.predict(sorted_df[["Temperature"]]) * USD_TO_INR
    sorted_df["Actual_Revenue"] = sorted_df["Revenue"] * USD_TO_INR

    st.subheader("📊 Interactive Visualizations")

    c1, c2 = st.columns(2)

    with c1:
        fig_scatter = px.scatter(
            df,
            x="Temperature",
            y=df["Revenue"] * USD_TO_INR,
            title="Temperature vs Revenue",
            labels={"Temperature": "Temperature (°C)", "y": "Revenue (₹)"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c2:
        fig_reg = go.Figure()
        fig_reg.add_trace(
            go.Scatter(
                x=sorted_df["Temperature"],
                y=sorted_df["Actual_Revenue"],
                mode="markers",
                name="Actual Revenue",
            )
        )
        fig_reg.add_trace(
            go.Scatter(
                x=sorted_df["Temperature"],
                y=sorted_df["Predicted_Revenue"],
                mode="lines",
                name="Regression Line",
            )
        )
        fig_reg.update_layout(
            title="Regression Fit",
            xaxis_title="Temperature (°C)",
            yaxis_title="Revenue (₹)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")

    st.subheader("🥧 Revenue Distribution by Temperature Range")

    # Categorize temperatures
    bins = [0, 15, 25, float('inf')]
    labels = ['Low (<15°C)', 'Medium (15-25°C)', 'High (>25°C)']
    df['Temp_Category'] = pd.cut(df['Temperature'], bins=bins, labels=labels)

    # Calculate revenue sum per category
    revenue_by_category = df.groupby('Temp_Category')['Revenue'].sum() * USD_TO_INR

    # Pie chart
    fig_pie = px.pie(
        values=revenue_by_category,
        names=revenue_by_category.index,
        title="Revenue Share by Temperature Range",
        template="plotly_dark",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Demand segmentation (count of records by temperature bucket)
    demand_counts = df["Temp_Category"].value_counts().reindex(labels, fill_value=0)
    fig_bar = px.bar(
        x=demand_counts.index,
        y=demand_counts.values,
        labels={"x": "Temperature Segment", "y": "Record Count"},
        title="Temperature Demand Segments",
        template="plotly_dark",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    st.subheader("🔍 Residual Analysis")

    # Calculate residuals
    sorted_df["Residuals"] = sorted_df["Actual_Revenue"] - sorted_df["Predicted_Revenue"]

    fig_residual = px.scatter(
        sorted_df,
        x="Temperature",
        y="Residuals",
        title="Residuals vs Temperature",
        labels={"Temperature": "Temperature (°C)", "Residuals": "Residuals (₹)"},
        template="plotly_dark",
    )
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residual, use_container_width=True)

    st.subheader("📈 Additional Analysis")

    c3, c4 = st.columns(2)

    with c3:
        fig_hist = px.histogram(
            df,
            x=df["Revenue"] * USD_TO_INR,
            nbins=20,
            title="Revenue Distribution",
            labels={"x": "Revenue (₹)"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        fig_line = px.line(
            sorted_df,
            x="Temperature",
            y="Predicted_Revenue",
            title="Predicted Revenue Trend by Temperature",
            labels={
                "Temperature": "Temperature (°C)",
                "Predicted_Revenue": "Predicted Revenue (₹)",
            },
            template="plotly_dark",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Report generation (downloadable HTML report)
    figures = {
        "forecast": fig_forecast,
        "scatter": fig_scatter,
        "regression": fig_reg,
        "residual": fig_residual,
        "hist": fig_hist,
        "trend": fig_line,
        "pie": fig_pie,
        "segment": fig_bar,
    }

    report_bytes = create_html_report(
        df=df,
        scenario_df=scenario_df,
        model_metrics=model_metrics,
        peak_temp=peak_temp,
        peak_revenue_inr=peak_revenue_inr,
        figures=figures,
    )

    st.download_button(
        "Download Sales Forecast Report",
        data=report_bytes,
        file_name="ice_cream_sales_report.html",
        mime="text/html",
        help="Download a complete report with charts, insights, and scenario analysis.",
    )

    if show_stats:
        st.markdown("---")
        st.subheader("📋 Summary Statistics")
        stats_df = df.copy()
        stats_df["Revenue"] = stats_df["Revenue"] * USD_TO_INR
        st.dataframe(stats_df.describe(), use_container_width=True)

    if show_data:
        st.markdown("---")
        st.subheader("🗂 Raw Dataset")
        data_df = df.copy()
        data_df["Revenue"] = data_df["Revenue"] * USD_TO_INR
        st.dataframe(data_df, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #9aa0a6; font-size: 14px;'>Powered by Machine Learning | Built with Streamlit</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()