"""
Conformalized Quantile Regression (CQR) (Romano, Patterson, & Candès, 2019).

Constructs adaptive, heteroscedastic prediction intervals via signed conformity
scores over conditional lower and upper pinball quantile estimates:
    E_i = max(q_{alpha/2}(X_i) - Y_i, Y_i - q_{1-alpha/2}(X_i))
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class QuantileIntervalResult:
    """Prediction intervals and coverage metrics."""
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    adjustment_offset: float
    empirical_coverage: Optional[float] = None
    mean_interval_width: Optional[float] = None


class ConformalQuantileRegressor:
    """
    Calibrates conditional quantile estimates using exact CQR signed conformity scores.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.calibrated_adjustment: Optional[float] = None

    def compute_conformity_scores(
        self,
        y_true: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> np.ndarray:
        """
        Computes signed conformity scores:
            E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
        Negative when Y_i is strictly inside [q_lo, q_hi], positive when outside.
        """
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        q_lo = np.asarray(q_lower, dtype=np.float64).ravel()
        q_hi = np.asarray(q_upper, dtype=np.float64).ravel()

        lower_err = q_lo - y_t
        upper_err = y_t - q_hi
        return np.maximum(lower_err, upper_err)

    def calibrate(
        self,
        y_true: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> float:
        """
        Calibrates the non-conformity adjustment quantile Q_{1-alpha}.
        """
        scores = self.compute_conformity_scores(y_true, q_lower, q_upper)
        n = len(scores)
        if n < 10:
            raise ValueError("Calibration set must contain at least 10 samples.")

        # Finite-sample quantile level
        q_level = min(1.0, np.ceil((n + 1.0) * (1.0 - self.alpha)) / float(n))
        self.calibrated_adjustment = float(np.quantile(scores, q_level, method="higher"))
        return self.calibrated_adjustment

    def predict(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies calibrated adjustment to construct valid prediction intervals:
            [q_lo - Q_{1-alpha}, q_hi + Q_{1-alpha}]
        """
        if self.calibrated_adjustment is None:
            raise RuntimeError("CQR model must be calibrated before generating predictions.")

        q_lo = np.asarray(q_lower, dtype=np.float64)
        q_hi = np.asarray(q_upper, dtype=np.float64)

        adjusted_lower = q_lo - self.calibrated_adjustment
        adjusted_upper = q_hi + self.calibrated_adjustment

        return adjusted_lower, adjusted_upper

    def evaluate(
        self,
        y_true: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> QuantileIntervalResult:
        """Evaluates prediction intervals against hold-out ground truth."""
        lower, upper = self.predict(q_lower, q_upper)
        y_t = np.asarray(y_true, dtype=np.float64)
        covered = (y_t >= lower) & (y_t <= upper)
        cov = float(np.mean(covered))
        width = float(np.mean(upper - lower))

        return QuantileIntervalResult(
            lower_bound=lower,
            upper_bound=upper,
            adjustment_offset=float(self.calibrated_adjustment),
            empirical_coverage=cov,
            mean_interval_width=width,
        )
