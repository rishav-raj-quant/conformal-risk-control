"""Unit tests for Online Adaptive Conformal Inference (Gibbs & Candès)."""

import numpy as np
import pytest
from conformal_control.adaptive_ci import OnlineAdaptiveConformalInference


def test_online_aci_drift_recovery():
    rng = np.random.default_rng(42)
    n_steps = 600

    # Step 1: Baseline regime (t < 300)
    # Step 2: Distribution shift / shock (t >= 300) with mean drift and moderate variance
    noise = np.concatenate([
        rng.normal(0, 1.0, size=300),
        rng.normal(0.8, 1.2, size=300),
    ])


    base_preds = np.zeros(n_steps)
    base_sigmas = np.ones(n_steps)
    calibration_residuals = np.abs(rng.normal(0, 1.0, size=500))

    target_alpha = 0.10
    aci = OnlineAdaptiveConformalInference(
        target_alpha=target_alpha,
        gamma_step=0.04,
        window_size=100,
    )

    history = aci.run_stream(
        y_stream=noise,
        base_predictions=base_preds,
        base_sigmas=base_sigmas,
        base_calibration_scores=calibration_residuals,
    )

    assert len(history) == n_steps
    # At shock point t=300, effective alpha drops (widening intervals to regain coverage)
    post_shock_alphas = [h.effective_alpha for h in history[310:450]]
    assert min(post_shock_alphas) < target_alpha

    # Long-term empirical coverage adapts towards target coverage
    post_adapt_errors = [h.error_indicator for h in history[400:]]
    post_adapt_err_rate = float(np.mean(post_adapt_errors))
    assert 0.04 <= post_adapt_err_rate <= 0.20

    overall_err_rate = float(np.mean([h.error_indicator for h in history]))
    assert 0.05 <= overall_err_rate <= 0.35

