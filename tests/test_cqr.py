"""Unit tests for Conformalized Quantile Regression (CQR)."""

import numpy as np
import pytest
from conformal_control.conformal_quantile import ConformalQuantileRegressor


def test_conformal_quantile_regression_coverage():
    rng = np.random.default_rng(77)
    n_cal = 500
    n_test = 800

    # Heteroscedastic noise: std depends on x
    x_cal = rng.uniform(0.5, 3.0, size=n_cal)
    sigma_cal = 0.5 * x_cal
    y_cal = 3.0 * x_cal + rng.normal(0, sigma_cal)
    # Synthetic quantile predictions (e.g. 5% and 95%)
    q_lo_cal = 3.0 * x_cal - 1.645 * sigma_cal
    q_hi_cal = 3.0 * x_cal + 1.645 * sigma_cal

    x_test = rng.uniform(0.5, 3.0, size=n_test)
    sigma_test = 0.5 * x_test
    y_test = 3.0 * x_test + rng.normal(0, sigma_test)
    q_lo_test = 3.0 * x_test - 1.645 * sigma_test
    q_hi_test = 3.0 * x_test + 1.645 * sigma_test

    cqr = ConformalQuantileRegressor(alpha=0.10)
    offset = cqr.calibrate(y_cal, q_lo_cal, q_hi_cal)

    eval_res = cqr.evaluate(y_test, q_lo_test, q_hi_test)
    assert eval_res.empirical_coverage is not None
    assert eval_res.empirical_coverage >= 0.88
    assert eval_res.mean_interval_width is not None
    assert eval_res.mean_interval_width > 0.0
