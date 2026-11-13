"""
reporting/report_generator.py
Generates a downloadable PDF forecasting report using ReportLab.
All content is derived from real model results — no dummy values.
"""
import io
import datetime
import pandas as pd
import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


PROJECT_NAME = "GridCast AI"
GROUP = "Group 14"
MEMBERS = ["Lawrence Sarkar", "Soumyadeep Nath", "Sayon De", "Pravin Agarwal"]


def generate_pdf(
    stats: dict,
    metrics: dict,
    forecast_df: pd.DataFrame,
    best_model_name: str,
) -> bytes:
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    stats : dict          Dataset statistics from DataProcessor.
    metrics : dict        {model_name: {MAE, RMSE, MAPE}} from ForecastEngine.
    forecast_df : pd.DataFrame  7-day forecast table (Date, Predicted_kWh).
    best_model_name : str Name of the best-performing model.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(
            "ReportLab is not installed. Run: pip install reportlab"
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18, textColor=colors.HexColor("#1e90ff"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#00d4ff"), spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)
    bold_body = ParagraphStyle("bold_body", parent=body, fontName="Helvetica-Bold")

    story = []

    # ── Title ─────────────────────────────────────────────────────────
    story.append(Paragraph(f"{PROJECT_NAME} — Forecasting Report", h1))
    story.append(Paragraph(f"{GROUP} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Paragraph(f"Team: {', '.join(MEMBERS)}", body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e90ff")))
    story.append(Spacer(1, 0.4 * cm))

    # ── Dataset Information ───────────────────────────────────────────
    story.append(Paragraph("1. Dataset Information", h2))
    ds_data = [
        ["Metric", "Value"],
        ["Records", str(stats.get("n_records", "—"))],
        ["Start Date", str(stats.get("start_date", "—"))],
        ["End Date", str(stats.get("end_date", "—"))],
        ["Total Consumption", f"{stats.get('total_kwh', 0):.2f} kWh"],
        ["Mean Consumption", f"{stats.get('mean_kwh', 0):.4f} kWh"],
        ["Max Consumption", f"{stats.get('max_kwh', 0):.4f} kWh"],
        ["Min Consumption", f"{stats.get('min_kwh', 0):.4f} kWh"],
    ]
    story.append(_make_table(ds_data))
    story.append(Spacer(1, 0.4 * cm))

    # ── Methodology ───────────────────────────────────────────────────
    story.append(Paragraph("2. Forecasting Methodology", h2))
    methodology = (
        "GridCast AI uses three time-series forecasting models: ARIMA (AutoRegressive Integrated Moving Average), "
        "SARIMA (Seasonal ARIMA), and LSTM (Long Short-Term Memory neural network). "
        "Data is split chronologically — no future data is used during training. "
        "Models are evaluated on a held-out test set using MAE, RMSE, and MAPE."
    )
    story.append(Paragraph(methodology, body))
    story.append(Spacer(1, 0.4 * cm))

    # ── Model Performance ─────────────────────────────────────────────
    story.append(Paragraph("3. Model Performance", h2))
    perf_data = [["Model", "MAE", "RMSE", "MAPE (%)"]]
    for model_name, m in metrics.items():
        mape_str = f"{m['MAPE']:.2f}" if m.get("MAPE") is not None else "N/A"
        perf_data.append([
            model_name,
            f"{m.get('MAE', 0):.4f}",
            f"{m.get('RMSE', 0):.4f}",
            mape_str,
        ])
    story.append(_make_table(perf_data))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"Best performing model (lowest MAE): <b>{best_model_name}</b>", body))
    story.append(Spacer(1, 0.4 * cm))

    # ── 7-Day Forecast ────────────────────────────────────────────────
    story.append(Paragraph(f"4. 7-Day Electricity Consumption Forecast ({best_model_name})", h2))
    fc_data = [["Date", "Predicted Consumption (kWh)"]]
    for _, row in forecast_df.iterrows():
        fc_data.append([str(row["Date"]), f"{row['Predicted_kWh']:.4f}"])
    # summary row
    fc_data.append(["Average", f"{forecast_df['Predicted_kWh'].mean():.4f}"])
    fc_data.append(["Peak", f"{forecast_df['Predicted_kWh'].max():.4f}"])
    story.append(_make_table(fc_data))
    story.append(Spacer(1, 0.4 * cm))

    # ── Conclusion ────────────────────────────────────────────────────
    story.append(Paragraph("5. Conclusion", h2))
    conclusion = (
        f"GridCast AI successfully trained and evaluated three forecasting models on the uploaded electricity "
        f"consumption dataset. The best-performing model based on Mean Absolute Error was {best_model_name}. "
        "The 7-day forecast is provided above for operational planning purposes. "
        "All predictions are derived from real model outputs on actual historical data."
    )
    story.append(Paragraph(conclusion, body))

    doc.build(story)
    return buffer.getvalue()


def _make_table(data: list) -> Table:
    """Build a styled ReportLab Table."""
    t = Table(data, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e90ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#0f1e2e"), colors.HexColor("#0a1628")]),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e0e6ed")),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a3a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ])
    t.setStyle(style)
    return t
