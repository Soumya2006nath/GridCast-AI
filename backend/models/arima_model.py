"""
models/arima_model.py
ARIMA forecasting model wrapper for GridCast AI.
Uses pmdarima auto_arima for automatic order selection, falling back
to a fixed order if the library is unavailable.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Optional


class ARIMAModel:
    """
    Wraps pmdarima's auto_arima for order selection and statsmodels
    ARIMA for fitting and prediction.
    """

    def __init__(self):
        self.model = None
        self.order = None
        self.fitted = False
        self.train_values: Optional[np.ndarray] = None

    def fit(self, train_series: pd.Series) -> None:
        """
        Fit ARIMA on the training series.
        Automatically selects (p, d, q) using auto_arima if available.
        """
        self.train_values = train_series.values.astype(float)

        try:
            import pmdarima as pm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auto = pm.auto_arima(
                    self.train_values,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    max_p=5, max_q=5, max_d=2,
                    information_criterion="aic",
                )
            self.order = auto.order
        except Exception:
            self.order = (2, 1, 2)  # sensible default

        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima = _ARIMA(self.train_values, order=self.order)
            self.model = arima.fit()
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        """Forecast `steps` future values."""
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        forecast = self.model.forecast(steps=steps)
        return np.asarray(forecast).ravel()

    def predict_in_sample(self, test_len: int) -> np.ndarray:
        """
        One-step-ahead rolling predictions over the test window.
        Uses the last in-sample forecast positions to avoid leakage.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_in_sample().")
        # get in-sample fitted values + out-of-sample
        n_train = len(self.train_values)
        pred = self.model.predict(start=n_train, end=n_train + test_len - 1)
        return np.asarray(pred).ravel()
