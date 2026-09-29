"""
Split Conformal Prediction for Regression and Multi-Class Classification.

Provides finite-sample, distribution-free marginal coverage guarantees:
    P(Y_{n+1} in C_{1-alpha}(X_{n+1})) >= 1 - alpha
under exchangeability.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np


@dataclass(frozen=True)
class ConformalMetrics:
    """Evaluation summary of conformal coverage and efficiency."""
    target_coverage: float
    empirical_coverage: float
    average_set_size: float
    coverage_error: float


class SplitConformalRegressor:
    """
    Inductive Split Conformal Regression.
    Calibrates prediction intervals around any base conditional mean predictor.
    Supports constant and locally normalized non-conformity scores.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.calibrated_quantile: Optional[float] = None
        self.normalized: bool = False

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_pred: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calibrates the non-conformity quantile Q_{1-alpha} on calibration split.
        Uses finite-sample correction: level = ceil((n + 1) * (1 - alpha)) / n.
        """
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_p = np.asarray(y_pred, dtype=np.float64).ravel()
        n = len(y_t)
        if n < 10:
            raise ValueError("Calibration set must contain at least 10 observations.")

        residuals = np.abs(y_t - y_p)
        if sigma_pred is not None:
            sigmas = np.asarray(sigma_pred, dtype=np.float64).ravel()
            sigmas = np.maximum(sigmas, 1e-6)
            scores = residuals / sigmas
            self.normalized = True
        else:
            scores = residuals
            self.normalized = False

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1.0) * (1.0 - self.alpha)) / float(n))
        self.calibrated_quantile = float(np.quantile(scores, q_level, method="higher"))
        return self.calibrated_quantile

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        sigma_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs prediction intervals [y_lower, y_upper].
        """
        if self.calibrated_quantile is None:
            raise RuntimeError("Model must be calibrated before predicting intervals.")

        y_p = np.asarray(y_pred, dtype=np.float64)
        if self.normalized:
            if sigma_pred is None:
                raise ValueError("sigma_pred required for normalized conformal prediction.")
            sigmas = np.asarray(sigma_pred, dtype=np.float64)
            spread = self.calibrated_quantile * sigmas
        else:
            spread = self.calibrated_quantile

        lower = y_p - spread
        upper = y_p + spread
        return lower, upper

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigma_pred: Optional[np.ndarray] = None,
    ) -> ConformalMetrics:
        """Computes empirical coverage and interval sharpness."""
        lower, upper = self.predict_intervals(y_pred, sigma_pred)
        y_t = np.asarray(y_true, dtype=np.float64)
        covered = (y_t >= lower) & (y_t <= upper)
        emp_cov = float(np.mean(covered))
        avg_width = float(np.mean(upper - lower))
        target_cov = 1.0 - self.alpha
        return ConformalMetrics(
            target_coverage=target_cov,
            empirical_coverage=emp_cov,
            average_set_size=avg_width,
            coverage_error=abs(emp_cov - target_cov),
        )


class AdaptivePredictionSetsClassifier:
    """
    Adaptive Prediction Sets (APS) for Multi-Class Classification (Romano et al., 2020).
    Constructs prediction sets C(X) subset of {1, ..., K} with exact marginal coverage.
    """

    def __init__(self, alpha: float = 0.10, randomized: bool = True) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.randomized = randomized
        self.calibrated_threshold: Optional[float] = None

    def _compute_conformity_scores(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Computes generalized APS non-conformity scores:
            E_i = sum_{pi(j) <= pi(y_i)} p_{pi(j)} - U_i * p_{y_i}
        """
        n, k = probs.shape
        scores = np.zeros(n, dtype=np.float64)
        rng = np.random.default_rng(seed)

        for i in range(n):
            p = probs[i]
            y = int(labels[i])
            sorted_indices = np.argsort(-p)
            sorted_p = p[sorted_indices]
            rank_of_y = int(np.where(sorted_indices == y)[0][0])

            cum_p = np.cumsum(sorted_p)
            score = cum_p[rank_of_y]
            if self.randomized:
                u = rng.uniform(0.0, 1.0)
                score -= u * sorted_p[rank_of_y]
            scores[i] = score

        return scores

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        seed: Optional[int] = 42,
    ) -> float:
        """Calibrates APS conformity score quantile."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=int).ravel()
        n = len(labels)

        scores = self._compute_conformity_scores(probs, labels, seed=seed)
        q_level = min(1.0, np.ceil((n + 1.0) * (1.0 - self.alpha)) / float(n))
        self.calibrated_threshold = float(np.quantile(scores, q_level, method="higher"))
        return self.calibrated_threshold

    def predict_sets(
        self,
        probs: np.ndarray,
        seed: Optional[int] = 42,
    ) -> List[List[int]]:
        """
        Returns prediction sets containing label indices for each observation.
        """
        if self.calibrated_threshold is None:
            raise RuntimeError("Classifier must be calibrated before generating prediction sets.")

        probs = np.asarray(probs, dtype=np.float64)
        n, k = probs.shape
        prediction_sets: List[List[int]] = []
        rng = np.random.default_rng(seed)

        for i in range(n):
            p = probs[i]
            sorted_indices = np.argsort(-p)
            sorted_p = p[sorted_indices]
            cum_p = np.cumsum(sorted_p)

            pred_set = []
            for rank in range(k):
                label = int(sorted_indices[rank])
                score = cum_p[rank]
                if self.randomized:
                    u = rng.uniform(0.0, 1.0)
                    score -= u * sorted_p[rank]

                pred_set.append(label)
                if score >= self.calibrated_threshold:
                    break

            prediction_sets.append(sorted(pred_set))

        return prediction_sets

    def evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        seed: Optional[int] = 42,
    ) -> ConformalMetrics:
        """Computes multi-class empirical coverage and set size."""
        sets = self.predict_sets(probs, seed=seed)
        labels = np.asarray(labels, dtype=int).ravel()
        covered = [labels[i] in sets[i] for i in range(len(labels))]
        emp_cov = float(np.mean(covered))
        avg_size = float(np.mean([len(s) for s in sets]))
        target_cov = 1.0 - self.alpha
        return ConformalMetrics(
            target_coverage=target_cov,
            empirical_coverage=emp_cov,
            average_set_size=avg_size,
            coverage_error=abs(emp_cov - target_cov),
        )
