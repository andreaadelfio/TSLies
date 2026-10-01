"""
Statistical core of TSLies: change detectors and their calibration.

This subpackage only depends on numpy and has no side effects at import time.
"""

from .focus import FocusResult, GaussianFocus, PoissonFocus, gaussian_focus, poisson_focus

__all__ = [
    "FocusResult",
    "GaussianFocus",
    "PoissonFocus",
    "gaussian_focus",
    "poisson_focus",
]
