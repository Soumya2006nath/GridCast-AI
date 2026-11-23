"""
models/sarima_model.py
SARIMA forecasting model wrapper for GridCast AI.
Captures seasonal patterns (daily/weekly) using seasonal order selection.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Optional


class SARIMAModel:
    """
    Fits a SARIMA(p,d,q)(P,D,Q)[m] model.
    Seasonal period m is inferred from the data frequency.
    """

    def __init__(self, seasonal_period: int = 24):
        """
        Parameters
        ----------
        seasonal_period : int
            Number of time steps per season.
            24 for hourly data (daily seasonality).
            7 for daily data (weekly seasonality).
        """
        self.seasonal_period = seasonal_period
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.fitted = False
        self.train_values: Optional[np.ndarray] = None

    def fit(self, train_series: pd.Series) -> None:
        """Fit SARIMA on the training series."""
        self.train_values = train_series.values.astype(float)
        m = self.seasonal_period

        try:
            import pmdarima as pm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auto = pm.auto_arima(
                    self.train_values,
                    seasonal=True,
                    m=m,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=3, max_q=3, max_d=1,
                    max_P=2, max_Q=2, max_D=1,
                    information_criterion="aic",
                )
            self.order = auto.order
            self.seasonal_order = auto.seasonal_order
        except Exception:
            # Sensible defaults for hourly data
            self.order = (1, 1, 1)
            self.seasonal_order = (1, 1, 0, m)

        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima = SARIMAX(
                self.train_values,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model = sarima.fit(disp=False, maxiter=100)
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        """Forecast `steps` future values."""
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        forecast = self.model.forecast(steps=steps)
        return np.asarray(forecast).ravel()

    def predict_in_sample(self, test_len: int) -> np.ndarray:
        """Out-of-sample predictions for the test window."""
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_in_sample().")
        n_train = len(self.train_values)
        pred = self.model.predict(start=n_train, end=n_train + test_len - 1)
        return np.asarray(pred).ravel()
