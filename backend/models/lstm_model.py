"""
backend/models/lstm_model.py
GPU-accelerated LSTM using Keras with TensorFlow backend.
Automatically uses RTX 4050 (or any CUDA GPU) if available.
Falls back to CPU transparently.
"""
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LSTMModel:
    def __init__(
        self,
        lookback: int = 48,
        units: int = 128,
        epochs: int = 30,
        batch_size: int = 64,
    ):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler_min: float = 0.0
        self.scaler_max: float = 1.0
        self.fitted = False
        self._last_window: Optional[np.ndarray] = None
        self._model = None

    # ------------------------------------------------------------------
    def fit(self, train_series: pd.Series) -> None:
        import tensorflow as tf
        self._log_device()

        values = train_series.values.astype(float)
        self.scaler_min = float(values.min())
        self.scaler_max = float(values.max())
        scaled = self._scale(values)

        X, y = self._make_sequences(scaled)
        if len(X) == 0:
            raise ValueError(f"Need at least {self.lookback + 1} records.")

        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Mixed-precision for speed on RTX series
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

        self._model = self._build(self.lookback, self.units)
        self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )],
        )
        self._last_window = scaled[-self.lookback:].copy()
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        window = self._last_window.copy()
        out = []
        for _ in range(steps):
            x = window[-self.lookback:].reshape(1, self.lookback, 1)
            p = float(self._model.predict(x, verbose=0)[0, 0])
            out.append(p)
            window = np.append(window, p)
        return self._inverse_scale(np.array(out))

    def predict_in_sample(self, train_series: pd.Series, test_len: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        scaled = self._scale(train_series.values.astype(float))
        window = scaled[-self.lookback:].copy()
        out = []
        for _ in range(test_len):
            x = window[-self.lookback:].reshape(1, self.lookback, 1)
            p = float(self._model.predict(x, verbose=0)[0, 0])
            out.append(p)
            window = np.append(window, p)
        return self._inverse_scale(np.array(out))

    def save(self, path: str) -> None:
        if self._model:
            self._model.save(path)

    def load(self, path: str) -> None:
        import tensorflow as tf
        self._model = tf.keras.models.load_model(path)
        self.fitted = True

    # ------------------------------------------------------------------
    @staticmethod
    def _build(lookback: int, units: int):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

        m = Sequential([
            LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units // 2, return_sequences=True),
            Dropout(0.2),
            LSTM(units // 4),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, dtype="float32"),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
        return m

    @staticmethod
    def _log_device():
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"[LSTM] Training on GPU: {gpus[0].name}")
            tf.config.experimental.set_memory_growth(gpus[0], True)
        else:
            print("[LSTM] No GPU detected — training on CPU.")

    def _make_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def _scale(self, arr):
        r = self.scaler_max - self.scaler_min
        return np.zeros_like(arr, dtype=float) if r == 0 else (arr - self.scaler_min) / r

    def _inverse_scale(self, arr):
        return arr * (self.scaler_max - self.scaler_min) + self.scaler_min
