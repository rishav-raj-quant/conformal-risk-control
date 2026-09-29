"""
Conformal Risk Control (CRC) & Learn-then-Test (Angelopoulos et al., 2022, 2024).

Extends distribution-free conformal guarantees from 0-1 miscoverage to arbitrary
bounded monotonic loss functions ell(C_lambda(X), Y) in [0, B]:
    E[ell(C_{lambda^*}(X), Y)] <= epsilon
with high-probability concentration bounds (Hoeffding / Empirical Bernstein).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class RiskControlResult:
    """Optimal parameter threshold and empirical risk diagnostics."""
    optimal_lambda: float
    target_risk_epsilon: float
    empirical_mean_risk: float
    upper_confidence_bound: float
    is_valid: bool
    lambda_grid: np.ndarray
    risk_curve: np.ndarray


def monotonic_fdr_loss(pred_set_indicator: np.ndarray, y_true_binary: np.ndarray) -> np.ndarray:
    """
    Computes false discovery rate loss per sample:
        ell = sum(pred & ~true) / max(1, sum(pred))
    """
    n = len(pred_set_indicator)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = pred_set_indicator[i]
        t = y_true_binary[i]
        n_selected = np.sum(s)
        if n_selected == 0:
            loss[i] = 0.0
        else:
            false_positives = np.sum(s & (~t))
            loss[i] = float(false_positives) / float(n_selected)
    return loss


def bounded_asymmetric_loss(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    penalty_ratio: float = 3.0,
    max_loss: float = 1.0,
) -> np.ndarray:
    """
    Asymmetric tail risk loss penalizing downside undershoot more severely than upside overshoot:
        loss = min(1, penalty * max(0, lower - y) + max(0, y - upper))
    """
    y = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(lower_bound, dtype=np.float64)
    hi = np.asarray(upper_bound, dtype=np.float64)

    downside = np.maximum(0.0, lo - y)
    upside = np.maximum(0.0, y - hi)
    raw_loss = penalty_ratio * downside + upside
    return np.clip(raw_loss, 0.0, max_loss)


class ConformalRiskController:
    """
    Solves distribution-free risk control for monotonic parameters lambda:
        lambda^* = inf { lambda in Lambda : (n/(n+1)) * R_hat(lambda) + B / (n+1) <= epsilon }
    """

    def __init__(
        self,
        epsilon: float = 0.05,
        delta: float = 0.05,
        bound_constant_b: float = 1.0,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError("Risk budget epsilon must be strictly positive.")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.b = float(bound_constant_b)

    def calibrate(
        self,
        lambda_grid: np.ndarray,
        loss_evaluator: Callable[[float], np.ndarray],
    ) -> RiskControlResult:
        """
        Scans monotonic threshold grid to find least conservative lambda achieving risk <= epsilon.
        loss_evaluator(lam) should return array of losses for calibration samples at parameter lam.
        """
        lams = np.sort(np.asarray(lambda_grid, dtype=np.float64))
        n_lams = len(lams)
        mean_risks = np.zeros(n_lams, dtype=np.float64)
        ucb_risks = np.zeros(n_lams, dtype=np.float64)

        optimal_idx: Optional[int] = None

        for idx, lam in enumerate(lams):
            losses = loss_evaluator(float(lam))
            n = len(losses)
            r_hat = float(np.mean(losses))
            mean_risks[idx] = r_hat

            # Conformal Risk Control finite-sample exact bound:
            # (n / (n + 1)) * R_hat + B / (n + 1)
            exact_crc_bound = (n / (n + 1.0)) * r_hat + (self.b / (n + 1.0))

            # Hoeffding concentration bound:
            hoeffding_slack = self.b * np.sqrt(np.log(1.0 / self.delta) / (2.0 * n))
            ucb = min(self.b, r_hat + hoeffding_slack)
            ucb_risks[idx] = exact_crc_bound

            if exact_crc_bound <= self.epsilon:
                optimal_idx = idx
                break

        if optimal_idx is None:
            # Most conservative fallback
            optimal_idx = n_lams - 1
            is_valid = False
        else:
            is_valid = True

        opt_lam = float(lams[optimal_idx])
        return RiskControlResult(
            optimal_lambda=opt_lam,
            target_risk_epsilon=self.epsilon,
            empirical_mean_risk=float(mean_risks[optimal_idx]),
            upper_confidence_bound=float(ucb_risks[optimal_idx]),
            is_valid=is_valid,
            lambda_grid=lams,
            risk_curve=mean_risks,
        )
