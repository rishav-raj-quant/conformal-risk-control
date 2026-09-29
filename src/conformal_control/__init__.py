"""
Conformal Risk Control: Distribution-Free Uncertainty Quantification & Online Adaptive Inference.
"""

from .split_conformal import (
    SplitConformalRegressor,
    AdaptivePredictionSetsClassifier,
    ConformalMetrics,
)
from .conformal_quantile import (
    ConformalQuantileRegressor,
    QuantileIntervalResult,
)
from .adaptive_ci import (
    OnlineAdaptiveConformalInference,
    AdaptiveCIRecord,
)
from .risk_control import (
    ConformalRiskController,
    RiskControlResult,
    monotonic_fdr_loss,
    bounded_asymmetric_loss,
)

__version__ = "0.1.0"
__all__ = [
    "SplitConformalRegressor",
    "AdaptivePredictionSetsClassifier",
    "ConformalMetrics",
    "ConformalQuantileRegressor",
    "QuantileIntervalResult",
    "OnlineAdaptiveConformalInference",
    "AdaptiveCIRecord",
    "ConformalRiskController",
    "RiskControlResult",
    "monotonic_fdr_loss",
    "bounded_asymmetric_loss",
]
