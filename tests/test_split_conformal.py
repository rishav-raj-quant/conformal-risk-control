"""Unit tests for Split Conformal Regression and Adaptive Prediction Sets."""

import numpy as np
import pytest
from conformal_control.split_conformal import (
    SplitConformalRegressor,
    AdaptivePredictionSetsClassifier,
)


def test_split_conformal_regression_coverage():
    rng = np.random.default_rng(42)
    # Generate linear calibration and test set with Gaussian noise
    x_cal = rng.uniform(-3, 3, size=500)
    y_cal = 2.0 * x_cal + rng.normal(0, 1.0, size=500)
    preds_cal = 2.0 * x_cal

    x_test = rng.uniform(-3, 3, size=1000)
    y_test = 2.0 * x_test + rng.normal(0, 1.0, size=1000)
    preds_test = 2.0 * x_test

    alpha = 0.10  # 90% target coverage
    cfr = SplitConformalRegressor(alpha=alpha)
    q_val = cfr.calibrate(y_true=y_cal, y_pred=preds_cal)

    metrics = cfr.evaluate(y_true=y_test, y_pred=preds_test)
    assert metrics.empirical_coverage >= 0.88
    assert metrics.coverage_error <= 0.03


def test_adaptive_prediction_sets_multiclass():
    rng = np.random.default_rng(101)
    n_cal = 600
    n_test = 1000
    n_classes = 4

    # Generate synthetic softmax probabilities
    logits_cal = rng.normal(0, 1, size=(n_cal, n_classes))
    probs_cal = np.exp(logits_cal) / np.sum(np.exp(logits_cal), axis=1, keepdims=True)
    labels_cal = np.array([rng.choice(n_classes, p=probs_cal[i]) for i in range(n_cal)])

    logits_test = rng.normal(0, 1, size=(n_test, n_classes))
    probs_test = np.exp(logits_test) / np.sum(np.exp(logits_test), axis=1, keepdims=True)
    labels_test = np.array([rng.choice(n_classes, p=probs_test[i]) for i in range(n_test)])

    aps = AdaptivePredictionSetsClassifier(alpha=0.15, randomized=True)
    aps.calibrate(probs=probs_cal, labels=labels_cal)

    metrics = aps.evaluate(probs=probs_test, labels=labels_test)
    assert metrics.empirical_coverage >= 0.83
    assert 1.0 <= metrics.average_set_size <= n_classes
