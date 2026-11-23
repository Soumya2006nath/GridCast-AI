"""
backend/forecasting/forecast_engine.py
Orchestrates model training, evaluation, and 7-day forecasting.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from evaluation.metrics import evaluate_all, best_model

FORECAST_DAYS = 7


class ForecastEngine:
    def __init__(self, resample_freq: str = "h"):
        self.resample_freq = resample_freq
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, dict] = {}
        self._forecasts: Dict[str, np.ndarray] = {}
        self._forecast_dates: Optional[pd.DatetimeIndex] = None
        self._train: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None
        self._lstm_train_trimmed = None

    def set_data(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        self._train = train
        self._test = test
        last_ts = test.index.max() if len(test) > 0 else train.index.max()
        freq = pd.tseries.frequencies.to_offset(self.resample_freq)
        self._forecast_dates = pd.date_range(
            start=last_ts + freq,
            periods=_steps_for_days(FORECAST_DAYS, self.resample_freq),
            freq=self.resample_freq,
        )

    def run_arima(self) -> dict:
        from models.arima_model import ARIMAModel
        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)
        m = ARIMAModel()
        m.fit(train_s)
        self._models["ARIMA"] = m
        metrics = evaluate_all(test_s.values, m.predict_in_sample(len(test_s)))
        self._results["ARIMA"] = metrics
        self._forecasts["ARIMA"] = np.maximum(0, m.predict(steps))
        return metrics

    def run_sarima(self) -> dict:
        from models.sarima_model import SARIMAModel
        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)
        m = SARIMAModel(seasonal_period=_seasonal_period(self.resample_freq))
        m.fit(train_s)
        self._models["SARIMA"] = m
        metrics = evaluate_all(test_s.values, m.predict_in_sample(len(test_s)))
        self._results["SARIMA"] = metrics
        self._forecasts["SARIMA"] = np.maximum(0, m.predict(steps))
        return metrics

    def run_lstm(self, lookback: int = 48, epochs: int = 30) -> dict:
        from models.lstm_model import LSTMModel
        train_s = self._train["consumption"]
        test_s = self._test["consumption"]
        steps = _steps_for_days(FORECAST_DAYS, self.resample_freq)
        m = LSTMModel(lookback=lookback, units=128, epochs=epochs, batch_size=64)
        m.fit(train_s)
        self._lstm_train_trimmed = train_s
        self._models["LSTM"] = m
        metrics = evaluate_all(test_s.values, m.predict_in_sample(train_s, len(test_s)))
        self._results["LSTM"] = metrics
        self._forecasts["LSTM"] = np.maximum(0, m.predict(steps))
        return metrics

    def get_forecast_df(self, model_name: str) -> pd.DataFrame:
        vals = self._forecasts[model_name]
        df = pd.DataFrame({
            "datetime": self._forecast_dates[: len(vals)],
            "predicted_kwh": vals,
        })
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date")["predicted_kwh"].sum().reset_index()
        daily.columns = ["Date", "Predicted_kWh"]
        return daily.head(FORECAST_DAYS)

    def get_all_metrics(self) -> Dict[str, dict]:
        return dict(self._results)

    def get_best_model(self) -> Optional[str]:
        return best_model(self._results) if self._results else None

    def get_forecast_summary(self, model_name: str) -> dict:
        df = self.get_forecast_df(model_name)
        return {
            "avg_daily_kwh": float(df["Predicted_kWh"].mean()),
            "peak_daily_kwh": float(df["Predicted_kWh"].max()),
            "total_7d_kwh": float(df["Predicted_kWh"].sum()),
        }

    def has_results(self) -> bool:
        return len(self._results) > 0


def _steps_for_days(days, freq):
    return {"h": days * 24, "D": days, "30min": days * 48, "15min": days * 96}.get(freq, days * 24)

def _seasonal_period(freq):
    return {"h": 24, "D": 7, "30min": 48, "15min": 96}.get(freq, 24)
