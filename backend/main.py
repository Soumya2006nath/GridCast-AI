"""
backend/main.py
GridCast AI — FastAPI backend.
Run: uvicorn main:app --reload --port 8000
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import json
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from preprocessing.data_processor import DataProcessor
from forecasting.forecast_engine import ForecastEngine
from utils.validation import validate_dataframe

app = FastAPI(title="GridCast AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state ───────────────────────────────────────────
_state: dict = {
    "processor": None,
    "engine": None,
    "loaded": False,
}


# ══════════════════════════════════════════════════════════════════════
# Dataset endpoints
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return _load_df(df)


@app.post("/api/dataset/sample")
def load_sample():
    path = os.path.join(os.path.dirname(__file__), "data", "sample_energy_data.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "Sample dataset not found.")
    df = pd.read_csv(path)
    return _load_df(df)


def _load_df(df: pd.DataFrame):
    val = validate_dataframe(df.copy())
    if not val["valid"]:
        raise HTTPException(400, detail={"errors": val["errors"]})

    proc = DataProcessor(resample_freq="h")
    processed = proc.process(df)
    train, test = proc.train_test_split(processed, test_days=30)

    engine = ForecastEngine(resample_freq="h")
    engine.set_data(train, test)

    _state["processor"] = proc
    _state["engine"] = engine
    _state["loaded"] = True

    return {
        "status": "ok",
        "warnings": val["warnings"],
        "stats": proc.stats,
        "records": len(processed),
    }


@app.get("/api/dataset/stats")
def get_stats():
    _require_data()
    proc: DataProcessor = _state["processor"]
    return proc.stats


@app.get("/api/dataset/preview")
def get_preview(rows: int = 100):
    _require_data()
    proc: DataProcessor = _state["processor"]
    df = proc.processed_df.head(rows).reset_index()
    return df.to_dict(orient="records")


@app.get("/api/dataset/daily")
def get_daily():
    _require_data()
    proc: DataProcessor = _state["processor"]
    df = proc.get_daily().reset_index()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/dataset/weekly")
def get_weekly():
    _require_data()
    proc: DataProcessor = _state["processor"]
    df = proc.get_weekly().reset_index()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/dataset/monthly")
def get_monthly():
    _require_data()
    proc: DataProcessor = _state["processor"]
    df = proc.get_monthly().reset_index()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/dataset/historical")
def get_historical(limit: int = 2000):
    _require_data()
    proc: DataProcessor = _state["processor"]
    df = proc.processed_df.tail(limit).reset_index()
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════
# Forecast endpoints
# ══════════════════════════════════════════════════════════════════════

class ForecastRequest(BaseModel):
    model: str          # "ARIMA" | "SARIMA" | "LSTM"
    lookback: int = 48
    epochs: int = 30


@app.post("/api/forecast/run")
def run_forecast(req: ForecastRequest):
    _require_data()
    engine: ForecastEngine = _state["engine"]
    model = req.model.upper()

    try:
        if model == "ARIMA":
            metrics = engine.run_arima()
        elif model == "SARIMA":
            metrics = engine.run_sarima()
        elif model == "LSTM":
            metrics = engine.run_lstm(lookback=req.lookback, epochs=req.epochs)
        else:
            raise HTTPException(400, f"Unknown model: {model}")
    except Exception as e:
        raise HTTPException(500, str(e))

    forecast_df = engine.get_forecast_df(model)
    forecast_df["Date"] = forecast_df["Date"].astype(str)
    summary = engine.get_forecast_summary(model)

    return {
        "model": model,
        "metrics": _clean_metrics(metrics),
        "forecast": forecast_df.to_dict(orient="records"),
        "summary": summary,
    }


@app.get("/api/forecast/{model_name}")
def get_forecast(model_name: str):
    _require_data()
    engine: ForecastEngine = _state["engine"]
    model = model_name.upper()
    if model not in engine._forecasts:
        raise HTTPException(404, f"Model '{model}' not yet run.")
    df = engine.get_forecast_df(model)
    df["Date"] = df["Date"].astype(str)
    return {
        "forecast": df.to_dict(orient="records"),
        "summary": engine.get_forecast_summary(model),
    }


# ══════════════════════════════════════════════════════════════════════
# Metrics endpoint
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/metrics")
def get_metrics():
    _require_data()
    engine: ForecastEngine = _state["engine"]
    if not engine.has_results():
        raise HTTPException(404, "No models trained yet.")
    raw = engine.get_all_metrics()
    cleaned = {k: _clean_metrics(v) for k, v in raw.items()}
    return {
        "metrics": cleaned,
        "best_model": engine.get_best_model(),
    }


# ══════════════════════════════════════════════════════════════════════
# Report endpoint
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/report/pdf")
def get_report():
    _require_data()
    engine: ForecastEngine = _state["engine"]
    proc: DataProcessor = _state["processor"]
    if not engine.has_results():
        raise HTTPException(400, "Train at least one model first.")

    best = engine.get_best_model()
    forecast_df = engine.get_forecast_df(best)
    forecast_df["Date"] = forecast_df["Date"].astype(str)

    from reporting.report_generator import generate_pdf
    pdf_bytes = generate_pdf(
        stats=proc.stats,
        metrics={k: _clean_metrics(v) for k, v in engine.get_all_metrics().items()},
        forecast_df=forecast_df,
        best_model_name=best,
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=GridCast_AI_Report.pdf"},
    )


# ══════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        gpu_info = [g.name for g in gpus]
    except Exception:
        gpu_info = []
    return {
        "status": "ok",
        "data_loaded": _state["loaded"],
        "gpus": gpu_info,
    }


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _require_data():
    if not _state["loaded"]:
        raise HTTPException(400, "No dataset loaded. POST /api/dataset/sample or /api/dataset/upload first.")


def _clean_metrics(m: dict) -> dict:
    """Replace None with null-safe float, convert numpy types."""
    return {
        k: (None if v is None else round(float(v), 6))
        for k, v in m.items()
    }
