"""
models/lstm_model.py
LSTM deep-learning forecasting model for GridCast AI.
Uses a sliding-window sequence approach with min-max normalisation.
No future information leaks into training.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


class LSTMModel:
    """
    LSTM sequence-to-one forecaster.
    Trains on (lookback,) -> (1,) windows.
    Forecasts future steps iteratively (rolling).
    """

    def __init__(self, lookback: int = 48, units: int = 64, epochs: int = 20, batch_size: int = 32):
        """
        Parameters
        ----------
        lookback : int   Number of past time steps fed as input.
        units : int      LSTM hidden units per layer.
        epochs : int     Training epochs.
        batch_size : int Mini-batch size.
        """
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_min: float = 0.0
        self.scaler_max: float = 1.0
        self.fitted = False
        self._last_window: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series) -> None:
        """Build and train the LSTM on the training series."""
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        values = train_series.values.astype(float)

        # Normalise to [0, 1] using ONLY training data statistics
        self.scaler_min = float(values.min())
        self.scaler_max = float(values.max())
        scaled = self._scale(values)

        X, y = self._make_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough data for lookback={self.lookback}. "
                f"Need at least {self.lookback + 1} records."
            )

        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model = self._build_model()
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Keep the last window for iterative forecasting
        self._last_window = scaled[-self.lookback:].copy()
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        """
        Iteratively forecast `steps` future values.
        Each new prediction is appended to the sliding window.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        window = self._last_window.copy()
        predictions = []
        for _ in range(steps):
            x = window[-self.lookback:].reshape(1, self.lookback, 1)
            pred_scaled = float(self.model.predict(x, verbose=0)[0, 0])
            predictions.append(pred_scaled)
            window = np.append(window, pred_scaled)

        return self._inverse_scale(np.array(predictions))

    def predict_in_sample(
        self, train_series: pd.Series, test_len: int
    ) -> np.ndarray:
        """
        One-step-ahead predictions over the test window using only
        past (training) data to seed the window — no leakage.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_in_sample().")

        values = train_series.values.astype(float)
        scaled = self._scale(values)
        window = scaled[-self.lookback:].copy()
        predictions = []
        for _ in range(test_len):
            x = window[-self.lookback:].reshape(1, self.lookback, 1)
            pred_scaled = float(self.model.predict(x, verbose=0)[0, 0])
            predictions.append(pred_scaled)
            window = np.append(window, pred_scaled)

        return self._inverse_scale(np.array(predictions))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(self.units // 2),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def _make_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def _scale(self, arr: np.ndarray) -> np.ndarray:
        rng = self.scaler_max - self.scaler_min
        if rng == 0:
            return np.zeros_like(arr)
        return (arr - self.scaler_min) / rng

    def _inverse_scale(self, arr: np.ndarray) -> np.ndarray:
        return arr * (self.scaler_max - self.scaler_min) + self.scaler_min
