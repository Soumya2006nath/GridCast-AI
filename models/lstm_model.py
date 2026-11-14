"""
models/lstm_model.py
LSTM forecasting model implemented in pure NumPy.
No TensorFlow/Keras — no DLL loading, no Application Control issues.
Uses a simple single-layer LSTM cell trained with BPTT (truncated).
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ── Activation helpers ────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def _tanh(x):
    return np.tanh(np.clip(x, -500, 500))


class LSTMModel:
    """
    Single-layer LSTM trained with Adam + BPTT, pure NumPy.
    API identical to the TensorFlow version so the rest of the
    codebase needs no changes.
    """

    def __init__(
        self,
        lookback: int = 48,
        units: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.scaler_min: float = 0.0
        self.scaler_max: float = 1.0
        self.fitted = False
        self._last_window: Optional[np.ndarray] = None

        # weights initialised in _init_weights
        self.Wf = self.Wi = self.Wc = self.Wo = None
        self.Uf = self.Ui = self.Uc = self.Uo = None
        self.bf = self.bi = self.bc = self.bo = None
        self.Wy = self.by = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_series: pd.Series) -> None:
        values = train_series.values.astype(float)
        self.scaler_min = float(values.min())
        self.scaler_max = float(values.max())
        scaled = self._scale(values)

        X, y = self._make_sequences(scaled)
        if len(X) == 0:
            raise ValueError(
                f"Not enough data for lookback={self.lookback}. "
                f"Need at least {self.lookback + 1} records."
            )

        self._init_weights()
        self._train(X, y)

        self._last_window = scaled[-self.lookback:].copy()
        self.fitted = True

    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")
        window = self._last_window.copy()
        predictions = []
        h = np.zeros((self.units,))
        c = np.zeros((self.units,))
        # warm up on window
        for t in range(len(window)):
            h, c = self._lstm_step(window[t], h, c)
        for _ in range(steps):
            pred = float(self.Wy.dot(h) + self.by)
            predictions.append(pred)
            h, c = self._lstm_step(pred, h, c)
        return self._inverse_scale(np.array(predictions))

    def predict_in_sample(
        self, train_series: pd.Series, test_len: int
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before predict_in_sample().")
        values = train_series.values.astype(float)
        scaled = self._scale(values)
        window = scaled[-self.lookback:]
        h = np.zeros((self.units,))
        c = np.zeros((self.units,))
        for t in range(len(window)):
            h, c = self._lstm_step(window[t], h, c)
        predictions = []
        for _ in range(test_len):
            pred = float(self.Wy.dot(h) + self.by)
            predictions.append(pred)
            h, c = self._lstm_step(pred, h, c)
        return self._inverse_scale(np.array(predictions))

    # ------------------------------------------------------------------
    # LSTM internals
    # ------------------------------------------------------------------

    def _init_weights(self):
        rng = np.random.default_rng(42)
        u = self.units
        scale = 0.1

        def W(): return rng.normal(0, scale, (u,))
        def U(): return rng.normal(0, scale, (u, u))
        def b(): return np.zeros((u,))

        self.Wf, self.Wi, self.Wc, self.Wo = W(), W(), W(), W()
        self.Uf, self.Ui, self.Uc, self.Uo = U(), U(), U(), U()
        self.bf, self.bi, self.bc, self.bo = b(), b(), b(), b()
        self.Wy = rng.normal(0, scale, (u,))
        self.by = np.zeros(1)

        # Adam moments
        self._m = {k: np.zeros_like(v) for k, v in self._param_dict().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._param_dict().items()}
        self._t = 0

    def _lstm_step(self, x_t: float, h: np.ndarray, c: np.ndarray):
        f = _sigmoid(self.Wf * x_t + self.Uf.dot(h) + self.bf)
        i = _sigmoid(self.Wi * x_t + self.Ui.dot(h) + self.bi)
        g = _tanh(self.Wc * x_t + self.Uc.dot(h) + self.bc)
        o = _sigmoid(self.Wo * x_t + self.Uo.dot(h) + self.bo)
        c_new = f * c + i * g
        h_new = o * _tanh(c_new)
        return h_new, c_new

    def _forward(self, x_seq: np.ndarray):
        """x_seq: (lookback,) -> scalar prediction"""
        h = np.zeros((self.units,))
        c = np.zeros((self.units,))
        for t in range(len(x_seq)):
            h, c = self._lstm_step(x_seq[t], h, c)
        y_pred = float(self.Wy.dot(h) + self.by)
        return y_pred, h

    def _train(self, X: np.ndarray, y: np.ndarray):
        """Mini-batch gradient descent with numerical gradients (simple BPTT approximation)."""
        eps = 1e-5
        n = len(X)
        param_keys = list(self._param_dict().keys())

        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                xb = X_shuf[start: start + self.batch_size]
                yb = y_shuf[start: start + self.batch_size]
                if len(xb) == 0:
                    continue

                # Compute loss
                preds = np.array([self._forward(x)[0] for x in xb])
                loss = float(np.mean((preds - yb) ** 2))
                epoch_loss += loss

                # Numerical gradients (small model — tractable)
                grads = {}
                params = self._param_dict()
                for k in param_keys:
                    p = params[k]
                    grad = np.zeros_like(p)
                    it = np.nditer(p, flags=["multi_index"])
                    while not it.finished:
                        ix = it.multi_index
                        orig = p[ix]
                        p[ix] = orig + eps
                        self._set_param(k, p)
                        preds_p = np.array([self._forward(x)[0] for x in xb])
                        loss_p = float(np.mean((preds_p - yb) ** 2))
                        p[ix] = orig - eps
                        self._set_param(k, p)
                        preds_m = np.array([self._forward(x)[0] for x in xb])
                        loss_m = float(np.mean((preds_m - yb) ** 2))
                        grad[ix] = (loss_p - loss_m) / (2 * eps)
                        p[ix] = orig
                        self._set_param(k, p)
                        it.iternext()
                    grads[k] = grad

                # Adam update
                self._t += 1
                b1, b2, ep = 0.9, 0.999, 1e-8
                for k in param_keys:
                    self._m[k] = b1 * self._m[k] + (1 - b1) * grads[k]
                    self._v[k] = b2 * self._v[k] + (1 - b2) * grads[k] ** 2
                    m_hat = self._m[k] / (1 - b1 ** self._t)
                    v_hat = self._v[k] / (1 - b2 ** self._t)
                    p = self._param_dict()[k]
                    p -= self.lr * m_hat / (np.sqrt(v_hat) + ep)
                    self._set_param(k, p)

    def _param_dict(self):
        return {
            "Wf": self.Wf, "Wi": self.Wi, "Wc": self.Wc, "Wo": self.Wo,
            "Uf": self.Uf, "Ui": self.Ui, "Uc": self.Uc, "Uo": self.Uo,
            "bf": self.bf, "bi": self.bi, "bc": self.bc, "bo": self.bo,
            "Wy": self.Wy, "by": self.by,
        }

    def _set_param(self, key: str, val: np.ndarray):
        setattr(self, key, val)

    # ------------------------------------------------------------------
    # Scaling helpers
    # ------------------------------------------------------------------

    def _scale(self, arr):
        rng = self.scaler_max - self.scaler_min
        if rng == 0:
            return np.zeros_like(arr)
        return (arr - self.scaler_min) / rng

    def _inverse_scale(self, arr):
        return arr * (self.scaler_max - self.scaler_min) + self.scaler_min

    # ------------------------------------------------------------------
    # Sequence builder
    # ------------------------------------------------------------------

    def _make_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
