"""
evaluation/metrics.py
Forecasting accuracy metrics for GridCast AI.
All metrics operate on actual observed values — no dummy scores.
"""
import numpy as np
import pandas as pd
from typing import Optional


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = _align(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = _align(actual, predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """
    Mean Absolute Percentage Error.
    Returns None if any actual value is zero (undefined).
    """
    actual, predicted = _align(actual, predicted)
    if np.any(actual == 0):
        # Use only non-zero actuals
        mask = actual != 0
        if mask.sum() == 0:
            return None
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def evaluate_all(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return MAE, RMSE, MAPE in a single dict."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
    }


def best_model(results: dict) -> str:
    """
    Given a dict {model_name: metrics_dict}, return the name
    of the model with the lowest MAE.
    """
    return min(results, key=lambda k: results[k].get("MAE", float("inf")))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _align(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(len(a), len(b))
    return a[:n], b[:n]
