"""Unit tests for Conformal Risk Control & Monotonic Loss Bounding."""

import numpy as np
import pytest
from conformal_control.risk_control import (
    ConformalRiskController,
    bounded_asymmetric_loss,
)


def test_conformal_risk_control_bound():
    rng = np.random.default_rng(99)
    n_cal = 400
    y_true = rng.normal(0, 1.0, size=n_cal)

    # Grid of candidate interval half-widths lambda
    lambda_grid = np.linspace(0.5, 4.0, 50)

    # Loss function evaluator: asymmetric downside risk
    def evaluator(lam: float):
        lo = -lam
        hi = lam
        return bounded_asymmetric_loss(y_true, lo, hi, penalty_ratio=2.0)

    epsilon = 0.08  # Tolerated expected risk <= 8%
    controller = ConformalRiskController(epsilon=epsilon, delta=0.05, bound_constant_b=1.0)
    res = controller.calibrate(lambda_grid=lambda_grid, loss_evaluator=evaluator)

    assert res.is_valid
    assert res.optimal_lambda > 0.5
    assert res.empirical_mean_risk <= epsilon
