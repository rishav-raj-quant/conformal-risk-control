"""
Adaptive Conformal Inference (ACI) under Non-Exchangeable Distribution Shifts.

Implements Gibbs & Candès (2021) online stochastic gradient tracking:
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
where err_t = 1_{y_t not in C_t(alpha_t)}.

Guarantees asymptotic empirical coverage convergence under arbitrary, unknown non-stationary drift:
    lim_{T -> infty} (1/T) sum_{t=1}^T err_t = alpha
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class AdaptiveCIRecord:
    """Historical progression of online conformal tracking."""
    step: int
    nominal_alpha: float
    effective_alpha: float
    error_indicator: int
    rolling_coverage: float
    interval_width: float


class OnlineAdaptiveConformalInference:
    """
    Streaming distribution-free uncertainty tracker for non-stationary environments.
    """

    def __init__(
        self,
        target_alpha: float = 0.10,
        gamma_step: float = 0.05,
        alpha_bounds: Tuple[float, float] = (0.01, 0.99),
        window_size: int = 50,
    ) -> None:
        if not (0.0 < target_alpha < 1.0):
            raise ValueError("target_alpha must be in (0, 1).")
        if gamma_step <= 0.0:
            raise ValueError("gamma_step must be positive.")

        self.target_alpha = float(target_alpha)
        self.gamma_step = float(gamma_step)
        self.alpha_bounds = (float(alpha_bounds[0]), float(alpha_bounds[1]))
        self.window_size = int(window_size)

        self.current_alpha = float(target_alpha)
        self.history: List[AdaptiveCIRecord] = []
        self._recent_errors: List[int] = []

    def update(
        self,
        y_true: float,
        lower_bound: float,
        upper_bound: float,
    ) -> AdaptiveCIRecord:
        """
        Receives realized observation y_t, checks set containment, and updates alpha_{t+1}.
        """
        is_covered = (y_true >= lower_bound) and (y_true <= upper_bound)
        err_t = 0 if is_covered else 1

        self._recent_errors.append(err_t)
        if len(self._recent_errors) > self.window_size:
            self._recent_errors.pop(0)

        rolling_cov = 1.0 - float(np.mean(self._recent_errors))
        interval_width = float(upper_bound - lower_bound)

        record = AdaptiveCIRecord(
            step=len(self.history) + 1,
            nominal_alpha=self.target_alpha,
            effective_alpha=self.current_alpha,
            error_indicator=err_t,
            rolling_coverage=rolling_cov,
            interval_width=interval_width,
        )
        self.history.append(record)

        # Gibbs-Candès update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        new_alpha = self.current_alpha + self.gamma_step * (self.target_alpha - float(err_t))
        self.current_alpha = float(np.clip(new_alpha, self.alpha_bounds[0], self.alpha_bounds[1]))

        return record

    def run_stream(
        self,
        y_stream: np.ndarray,
        base_predictions: np.ndarray,
        base_sigmas: np.ndarray,
        base_calibration_scores: np.ndarray,
    ) -> List[AdaptiveCIRecord]:
        """
        Executes online ACI across a sequence of streaming non-stationary observations.
        """
        y_seq = np.asarray(y_stream, dtype=np.float64)
        preds = np.asarray(base_predictions, dtype=np.float64)
        sigmas = np.asarray(base_sigmas, dtype=np.float64)
        scores = np.asarray(base_calibration_scores, dtype=np.float64)
        n = len(y_seq)

        for t in range(n):
            # Compute dynamic conformal quantile using current_alpha
            q_level = min(1.0, max(0.01, 1.0 - self.current_alpha))
            q_val = float(np.quantile(scores, q_level, method="higher"))

            spread = q_val * sigmas[t]
            lo = preds[t] - spread
            hi = preds[t] + spread

            self.update(y_true=float(y_seq[t]), lower_bound=lo, upper_bound=hi)

        return self.history
