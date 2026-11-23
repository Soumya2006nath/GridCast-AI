"""
backend/reporting/report_generator.py
Generates a downloadable PDF forecasting report using ReportLab.
"""
import io
import datetime
import pandas as pd

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


def generate_pdf(stats, metrics, forecast_df, best_model_name) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab not installed. Run: pip install reportlab")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18,
                        textColor=colors.HexColor("#1e90ff"), spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#00d4ff"), spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)

    story = []
    story.append(Paragraph(f"{PROJECT_NAME} — Forecasting Report", h1))
    story.append(Paragraph(
        f"{GROUP} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Team: {', '.join(MEMBERS)}", body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e90ff")))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("1. Dataset Information", h2))
    ds_data = [["Metric", "Value"],
               ["Records", str(stats.get("n_records", "—"))],
               ["Start Date", str(stats.get("start_date", "—"))],
               ["End Date", str(stats.get("end_date", "—"))],
               ["Total Consumption", f"{stats.get('total_kwh', 0):.2f} kWh"],
               ["Mean Consumption", f"{stats.get('mean_kwh', 0):.4f} kWh"],
               ["Max Consumption", f"{stats.get('max_kwh', 0):.4f} kWh"],
               ["Min Consumption", f"{stats.get('min_kwh', 0):.4f} kWh"]]
    story.append(_table(ds_data))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2. Forecasting Methodology", h2))
    story.append(Paragraph(
        "GridCast AI uses ARIMA, SARIMA, and an LSTM neural network (GPU-accelerated). "
        "Data is split chronologically — no future data is used during training. "
        "Models are evaluated on a held-out 30-day test set.", body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("3. Model Performance", h2))
    perf = [["Model", "MAE", "RMSE", "MAPE (%)"]]
    for mn, mv in metrics.items():
        perf.append([mn,
                     f"{mv.get('MAE', 0):.4f}",
                     f"{mv.get('RMSE', 0):.4f}",
                     f"{mv['MAPE']:.2f}" if mv.get("MAPE") is not None else "N/A"])
    story.append(_table(perf))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(f"Best model (lowest MAE): <b>{best_model_name}</b>", body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(f"4. 7-Day Forecast ({best_model_name})", h2))
    fc = [["Date", "Predicted Consumption (kWh)"]]
    for _, row in forecast_df.iterrows():
        fc.append([str(row["Date"]), f"{row['Predicted_kWh']:.4f}"])
    fc.append(["Average", f"{forecast_df['Predicted_kWh'].mean():.4f}"])
    fc.append(["Peak", f"{forecast_df['Predicted_kWh'].max():.4f}"])
    story.append(_table(fc))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("5. Conclusion", h2))
    story.append(Paragraph(
        f"GridCast AI trained and evaluated three models. Best model by MAE: {best_model_name}. "
        "The 7-day forecast above is derived from real model outputs on actual historical data.", body))

    doc.build(story)
    return buffer.getvalue()


def _table(data):
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e90ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#0f1e2e"), colors.HexColor("#0a1628")]),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#e0e6ed")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#1a2a3a")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t
