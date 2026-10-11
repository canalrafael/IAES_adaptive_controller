"""
Auto-Regressive Integrated Moving Average (ARIMA) Model for SPViT
=================================================================
Implements the ARIMA(p, d, q) predictor used in SPViT (Algorithm 1)
to forecast computing capacity (FLOPS) and network bandwidth (BW)
across heterogeneous edge-server devices.
"""

from typing import List, Optional
import numpy as np


class ARIMA:
    """
    ARIMA(p, d, q) predictor tailored for online edge-device metric forecasting.
    """

    def __init__(self, p: int = 5, d: int = 1, q: int = 0):
        self.p = p
        self.d = d
        self.q = q
        self.data: List[float] = []
        self.history: List[float] = []
        self.coefficients: List[float] = []

    def difference(self, series: List[float], order: int = 1) -> List[float]:
        """Calculates k-th order differences of a series."""
        diff = list(series)
        for _ in range(order):
            if len(diff) < 2:
                return [0.0]
            diff = [diff[i] - diff[i - 1] for i in range(1, len(diff))]
        return diff

    def inverse_difference(self, last_raw_value: float, diff_val: float) -> float:
        """Inverts a 1st order difference back to original scale."""
        return last_raw_value + diff_val

    def fit(self, series: List[float]) -> "ARIMA":
        """
        Fits the AR parameters on the differenced data series.
        """
        self.data = [float(x) for x in series]
        self.history = list(self.data)

        if len(self.data) <= self.p + self.d:
            # Insufficient samples for full regression; default to mean or last observation
            self.coefficients = [1.0 / max(1, self.p)] * self.p
            return self

        # Apply order-d differencing
        diff_data = self.difference(self.data, self.d)

        if len(diff_data) <= self.p:
            self.coefficients = [1.0 / max(1, self.p)] * self.p
            return self

        # Construct autoregression matrix: X * w = y
        X = []
        y = []
        for i in range(self.p, len(diff_data)):
            X.append(diff_data[i - self.p:i][::-1])  # lag features [t-1, t-2, ...]
            y.append(diff_data[i])

        X_mat = np.array(X, dtype=np.float64)
        y_vec = np.array(y, dtype=np.float64)

        try:
            # Solve regularized least-squares for AR coefficients
            reg = 1e-4 * np.eye(self.p)
            coeffs, _, _, _ = np.linalg.lstsq(
                X_mat.T @ X_mat + reg, X_mat.T @ y_vec, rcond=None
            )
            self.coefficients = coeffs.tolist()
        except Exception:
            self.coefficients = [1.0 / self.p] * self.p

        return self

    def predict_next(self, default_val: Optional[float] = None) -> float:
        """
        Predicts the metric value (FLOPS or BW) for the next minibatch (n+1).
        """
        if not self.data:
            return default_val if default_val is not None else 1.0

        if len(self.data) < 2:
            return float(self.data[-1])

        diff_data = self.difference(self.data, self.d)

        if len(diff_data) < self.p:
            # If not enough history, use exponentially weighted moving average
            return float(np.mean(self.data[-min(len(self.data), 3):]))

        recent_lags = np.array(diff_data[-self.p:][::-1], dtype=np.float64)
        predicted_diff = float(np.dot(recent_lags, np.array(self.coefficients)))

        # Invert difference
        predicted_val = self.inverse_difference(self.data[-1], predicted_diff)
        # Ensure positive metric
        return max(1e-4, predicted_val)
