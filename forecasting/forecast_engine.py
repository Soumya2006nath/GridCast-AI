"""
forecasting/forecast_engine.py
Orchestrates model training, evaluation, and 7-day forecasting.
Keeps preprocessing, training, and evaluation in separate concerns.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from evaluation.metrics import evaluate_all, best_model


FORECAST_DAYS = 7


class ForecastEngine:
    """
    High-level interface used by the Streamlit dashboard.
    Owns model instances and results caching.
    """

    def __init__(self, resample_freq: str = "h"):
        self.resample_freq = resample_freq
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, dict] = {}
        self._forecasts: Dict[str, np.ndarray] = {}
        self._forecast_dates: Optional[pd.DatetimeIndex] = None
        self._train: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_data(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """Provide already-split train/test DataFrames (must have 'consumption' column)."""
        self._train = train
        self._test = test
        last_ts = test.index.max() if len(test) > 0 else train.index.max()
        freq = pd.tseries.frequencies.to_offset(self.resample_freq)
        self._forecast_dates = pd.date_range(
            start=last_ts + freq,
            periods=_steps_for_days(FORECAST_DAYS, self.resample_freq),
            freq=self.resample_freq,
        )

    # ------------------------------------------------------------------
    # Individual model runners
    # ------------------------------------------------------------------

    def run_arima(self, progress_cb=None) -> dict:
        """Train ARIMA, evaluate on test set, generate 7-day forecast."""
        from models.arima_model import ARIMAModel

        if self._train is None:
            raise RuntimeError("Call set_data() first.")

        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)

        if progress_cb:
            progress_cb("Fitting ARIMA…")

        m = ARIMAModel()
        m.fit(train_s)
        self._models["ARIMA"] = m

        test_preds = m.predict_in_sample(len(test_s))
        metrics = evaluate_all(test_s.values, test_preds)
        self._results["ARIMA"] = metrics

        forecast = np.maximum(0, m.predict(steps))
        self._forecasts["ARIMA"] = forecast

        return metrics

    def run_sarima(self, progress_cb=None) -> dict:
        """Train SARIMA, evaluate on test set, generate 7-day forecast."""
        from models.sarima_model import SARIMAModel

        if self._train is None:
            raise RuntimeError("Call set_data() first.")

        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)
        m_period = _seasonal_period(self.resample_freq)

        if progress_cb:
            progress_cb("Fitting SARIMA…")

        m = SARIMAModel(seasonal_period=m_period)
        m.fit(train_s)
        self._models["SARIMA"] = m

        test_preds = m.predict_in_sample(len(test_s))
        metrics = evaluate_all(test_s.values, test_preds)
        self._results["SARIMA"] = metrics

        forecast = np.maximum(0, m.predict(steps))
        self._forecasts["SARIMA"] = forecast

        return metrics

    def run_lstm(self, lookback: int = 48, epochs: int = 20, progress_cb=None) -> dict:
        """Train LSTM, evaluate on test set, generate 7-day forecast."""
        from models.lstm_model import LSTMModel

        if self._train is None:
            raise RuntimeError("Call set_data() first.")

        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)

        if progress_cb:
            progress_cb("Training LSTM (this may take a minute)…")

        from models.lstm_model import _try_keras
        if _try_keras():
            # Full Keras LSTM — use all training data
            train_trimmed = train_s
            lb = lookback
            u = 64
        else:
            # NumPy ridge fallback — trim to keep it fast
            train_trimmed = train_s.iloc[-500:] if len(train_s) > 500 else train_s
            lb = min(lookback, max(4, len(train_trimmed) // 4))
            u = 16

        m = LSTMModel(lookback=lb, units=u, epochs=epochs)
        m.fit(train_trimmed)
        self._lstm_train_trimmed = train_trimmed
        self._models["LSTM"] = m

        test_preds = m.predict_in_sample(train_trimmed, len(test_s))
        metrics = evaluate_all(test_s.values, test_preds)
        self._results["LSTM"] = metrics

        forecast = np.maximum(0, m.predict(steps))
        self._forecasts["LSTM"] = forecast

        return metrics

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_forecast_df(self, model_name: str) -> pd.DataFrame:
        """Return a DataFrame with date and predicted consumption."""
        if model_name not in self._forecasts:
            raise ValueError(f"Model '{model_name}' has not been run.")
        vals = self._forecasts[model_name]
        # Aggregate to daily totals for the 7-day view
        df = pd.DataFrame({
            "datetime": self._forecast_dates[: len(vals)],
            "predicted_kwh": vals,
        })
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["predicted_kwh"].sum().reset_index()
        daily.columns = ["Date", "Predicted_kWh"]
        daily = daily.head(FORECAST_DAYS)
        return daily

    def get_all_metrics(self) -> Dict[str, dict]:
        return dict(self._results)

    def get_best_model(self) -> Optional[str]:
        if not self._results:
            return None
        return best_model(self._results)

    def get_forecast_summary(self, model_name: str) -> dict:
        df = self.get_forecast_df(model_name)
        return {
            "avg_daily_kwh": float(df["Predicted_kWh"].mean()),
            "peak_daily_kwh": float(df["Predicted_kWh"].max()),
            "total_7d_kwh": float(df["Predicted_kWh"].sum()),
        }

    def has_results(self) -> bool:
        return len(self._results) > 0


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _steps_for_days(days: int, freq: str) -> int:
    """Convert days to number of steps for a given frequency."""
    mapping = {
        "h": days * 24,
        "D": days,
        "30min": days * 48,
        "15min": days * 96,
    }
    return mapping.get(freq, days * 24)


def _seasonal_period(freq: str) -> int:
    """Return the natural seasonal period for a frequency."""
    mapping = {
        "h": 24,
        "D": 7,
        "30min": 48,
        "15min": 96,
    }
    return mapping.get(freq, 24)
