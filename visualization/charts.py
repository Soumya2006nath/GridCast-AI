"""
visualization/charts.py
All Plotly chart builders for GridCast AI dashboard.
Every chart uses real data passed in — no hardcoded values.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── Theme colours ────────────────────────────────────────────────────
NAVY = "#0d1b2a"
BLUE = "#1e90ff"
CYAN = "#00d4ff"
GREEN = "#00e676"
ORANGE = "#ff9100"
PURPLE = "#9c27b0"
GREY = "#90a4ae"
BG = "#0a1628"
CARD_BG = "#0f2035"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD_BG,
    font=dict(color="#e0e6ed", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a2a3a", zeroline=False),
    yaxis=dict(gridcolor="#1a2a3a", zeroline=False),
)


def historical_line(df: pd.DataFrame, title: str = "Historical Electricity Consumption") -> go.Figure:
    """Full historical consumption line chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["consumption"],
        mode="lines",
        name="Consumption (kWh)",
        line=dict(color=CYAN, width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(color=CYAN, size=15)),
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        hovermode="x unified",
    )
    return fig


def daily_pattern(daily_df: pd.DataFrame) -> go.Figure:
    """Daily consumption bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_df.index,
        y=daily_df["consumption"],
        name="Daily kWh",
        marker_color=BLUE,
        marker_opacity=0.8,
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Daily Consumption", font=dict(color=BLUE, size=15)),
        xaxis_title="Date",
        yaxis_title="Daily Consumption (kWh)",
        bargap=0.1,
    )
    return fig


def weekly_trend(weekly_df: pd.DataFrame) -> go.Figure:
    """Weekly trend line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly_df.index, y=weekly_df["consumption"],
        mode="lines+markers",
        name="Weekly kWh",
        line=dict(color=GREEN, width=2),
        marker=dict(size=5, color=GREEN),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Weekly Consumption Trend", font=dict(color=GREEN, size=15)),
        xaxis_title="Week",
        yaxis_title="Weekly Consumption (kWh)",
    )
    return fig


def monthly_trend(monthly_df: pd.DataFrame) -> go.Figure:
    """Monthly bar chart."""
    labels = [str(d.strftime("%b %Y")) for d in monthly_df.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=monthly_df["consumption"],
        name="Monthly kWh",
        marker_color=PURPLE,
        marker_opacity=0.85,
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Monthly Consumption", font=dict(color=PURPLE, size=15)),
        xaxis_title="Month",
        yaxis_title="Monthly Consumption (kWh)",
        bargap=0.15,
    )
    return fig


def consumption_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of consumption values."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["consumption"],
        nbinsx=60,
        name="Consumption",
        marker_color=ORANGE,
        opacity=0.8,
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Consumption Distribution", font=dict(color=ORANGE, size=15)),
        xaxis_title="Consumption (kWh)",
        yaxis_title="Frequency",
    )
    return fig


def forecast_chart(forecast_df: pd.DataFrame, historical_avg: float, model_name: str) -> go.Figure:
    """7-day forecast bar chart with historical average reference line."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(d) for d in forecast_df["Date"]],
        y=forecast_df["Predicted_kWh"],
        name="Forecast",
        marker_color=CYAN,
        marker_opacity=0.85,
    ))
    fig.add_hline(
        y=historical_avg,
        line_dash="dash",
        line_color=ORANGE,
        annotation_text=f"Historical avg: {historical_avg:.2f} kWh",
        annotation_font_color=ORANGE,
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"7-Day Forecast — {model_name}", font=dict(color=CYAN, size=15)),
        xaxis_title="Date",
        yaxis_title="Predicted Consumption (kWh)",
        bargap=0.2,
    )
    return fig


def model_comparison_bar(metrics: dict) -> go.Figure:
    """Side-by-side MAE / RMSE / MAPE comparison across models."""
    models = list(metrics.keys())
    mae_vals = [metrics[m].get("MAE", 0) for m in models]
    rmse_vals = [metrics[m].get("RMSE", 0) for m in models]
    mape_vals = [v if (v := metrics[m].get("MAPE")) is not None else 0 for m in models]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE", "RMSE", "MAPE (%)"],
    )
    colors = [BLUE, CYAN, GREEN]

    for col, (vals, label, color) in enumerate(
        [(mae_vals, "MAE", BLUE), (rmse_vals, "RMSE", CYAN), (mape_vals, "MAPE", GREEN)],
        start=1,
    ):
        fig.add_trace(
            go.Bar(x=models, y=vals, name=label, marker_color=color, showlegend=False),
            row=1, col=col,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Model Performance Comparison", font=dict(color=GREY, size=15)),
        height=380,
    )
    return fig


def actual_vs_predicted(test_series: pd.Series, predictions: np.ndarray, model_name: str) -> go.Figure:
    """Overlay actual vs predicted on the test window."""
    n = min(len(test_series), len(predictions))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_series.index[:n], y=test_series.values[:n],
        mode="lines", name="Actual",
        line=dict(color=GREEN, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=test_series.index[:n], y=predictions[:n],
        mode="lines", name=f"Predicted ({model_name})",
        line=dict(color=ORANGE, width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"Actual vs Predicted — {model_name}", font=dict(color=GREY, size=15)),
        xaxis_title="Date",
        yaxis_title="Consumption (kWh)",
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig
