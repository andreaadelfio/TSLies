"""
Exact FOCuS detectors for changes in a Gaussian mean or a Poisson rate.

At every time ``t`` the detectors compute the log-likelihood ratio (LLR) of

- H0: no change since the last reset, against
- H1: a change at some unknown ``tau < t``, after which the mean (Gaussian) or the
  rate (Poisson) is larger than under H0 by at least a minimum intensity ``mu_min``,

maximised over both ``tau`` and the post-change parameter:

- Gaussian, ``x_s ~ N(0, 1)`` under H0 and ``N(mu, 1)`` with ``mu >= mu_min >= 0`` after the
  change: ``LLR_t = max_tau max_mu (mu S - n mu**2 / 2)``, with ``n = t - tau`` and
  ``S = sum_{tau < s <= t} x_s``. For ``mu_min = 0`` this is ``max_tau S**2 / (2 n)``, ``S > 0``.
- Poisson, ``k_s ~ Pois(lambda_s)`` under H0 and ``Pois(mu * lambda_s)`` with ``mu >= mu_min >= 1``
  after the change: ``LLR_t = max_tau max_mu (K log(mu) - L (mu - 1))``, with ``K = sum k_s`` and
  ``L = sum lambda_s``. For ``mu_min = 1`` this is ``max_tau K log(K / L) - (K - L)``, ``K > L``.

The significance is reported as ``sqrt(2 * LLR)``. For a *fixed* segment and ``mu_min`` at its
default this is the z-score ``S / sqrt(n)`` (Gaussian) or the signed-root likelihood ratio
(Poisson), but the maximisation over ``tau`` makes its null distribution heavier-tailed than
N(0, 1): a threshold on it must be calibrated for a target false-alarm rate, it cannot be read as
a number of sigmas.

FOCuS, as defined in the references below, is a stopping rule: it runs until its statistic first
exceeds a threshold, reports the change, and stops. To monitor a continuous stream, the batch
functions accept ``restart_above``: the detector then starts again from scratch right after every
sample whose significance exceeds it. Without restarts, the evidence of a strong change is never
forgotten: after the change has ended its significance decreases only like ``1 / sqrt(time)``, so
it can stay over threshold for hours and hide the changes that follow. With restarts a change
raises an alarm and is then forgotten; a long change raises a sequence of alarms, one after the
other, while it lasts.

Alternatively, ``background_above`` keeps the memory after an alarm, but stores the sample that
raised it as background (``x = 0``, or ``k = lambda``) instead of its value, so that the evidence
cannot grow much above the threshold. A long change then raises an alarm at almost every sample
while it lasts. But the evidence of past changes is never emptied, only kept just below the
threshold, where it fades like ``1 / sqrt(time)``: alarms can keep coming after a change has ended,
and on long stretches without resets changes hours apart are joined into one. On synthetic data
both find the same changes, and ``restart_above`` is the safer choice (see
``tslies/examples/example4/trigger_validation.ipynb`` and ``benchmarks/long_benchmark.py``).

``mu_min`` restricts the test to changes of at least that intensity. A persistent bias of the
residuals smaller than ``mu_min / 2`` (Gaussian) then never accumulates evidence, however long it
lasts, which protects the detector against slow background mismatch, and it bounds the number of
candidate changepoints kept in memory.

Algorithm
---------
For a fixed post-change parameter, the LLR of a changepoint ``tau`` equals, up to terms that do
not depend on ``tau``, ``-(v_tau - s * u_tau)`` with

- Gaussian: ``u_tau = tau``, ``v_tau = sum_{s <= tau} x_s`` and ``s = mu / 2``;
- Poisson: ``u_tau = sum_{s <= tau} lambda_s``, ``v_tau = sum_{s <= tau} (k_s - lambda_s)`` and
  ``s = (mu - 1) / log(mu) - 1``.

Minimising a linear function over the points ``(u_tau, v_tau)`` selects a vertex of their lower
convex hull, and the slopes ``s`` allowed by ``mu >= mu_min`` only reach the vertices from the one
where the hull slope crosses ``s(mu_min)`` rightwards. Keeping exactly those vertices as candidate
changepoints is therefore lossless. Updating them costs amortised O(1) per sample, plus a scan of
the candidates, whose number grows like ``log(n)`` under H0 when ``mu_min`` is at its default and
stays bounded otherwise. This is the functional pruning of FOCuS expressed geometrically.

References
----------
- G. Romano, I. A. Eckley, P. Fearnhead, G. Rigaill, "Fast online changepoint detection via
  functional pruning CUSUM statistics", JMLR 24 (2023), arXiv:2110.08205.
- K. Ward, G. Dilillo, I. Eckley, P. Fearnhead, "Poisson-FOCuS: an efficient online method for
  detecting count bursts with application to gamma ray burst detection", arXiv:2208.01494.
  Reference notebooks: https://github.com/kesward/FOCuS
- R. Crupi, G. Dilillo, K. Ward, E. Bissaldi, F. Fiore, A. Vacchi, "Searching for long faint
  astronomical high energy transients: a data driven approach", Experimental Astronomy (2023),
  doi:10.1007/s10686-023-09915-7. Code: https://github.com/rcrupi/DeepGRB, whose Poisson-FOCuS
  trigger introduced the ``mu_min`` cut, implemented here exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from math import isfinite, log

import numpy as np

__all__ = [
    "FocusResult",
    "GaussianFocus",
    "PoissonFocus",
    "gaussian_focus",
    "poisson_focus",
]


class _HullFocus:
    """Candidate-changepoint bookkeeping shared by the Gaussian and Poisson detectors."""

    def __init__(self, mu_min: float, slope_min: float) -> None:
        self.mu_min = mu_min
        self._slope_min = slope_min
        self.reset()

    def reset(self) -> None:
        """
        Forget all the data seen so far, e.g. after a gap in the time series.

        Returns
        -------
        - None
        """
        self._n = 0
        self._u = 0.0
        self._v = 0.0
        # candidate changepoints as (u, v, number of samples seen at that point)
        self._hull = deque([(0.0, 0.0, 0)])

    def _state(self) -> tuple:
        """A copy of what the detector remembers, to go back to it with :meth:`_restore`."""
        return self._n, self._u, self._v, deque(self._hull)

    def _restore(self, state: tuple) -> None:
        self._n, self._u, self._v, self._hull = state

    def _segment_llr(self, du: float, dv: float) -> float:
        raise NotImplementedError

    def _push(self, du: float, dv: float) -> tuple[float, int]:
        self._n += 1
        self._u += du
        self._v += dv
        u, v, n = self._u, self._v, self._n
        hull = self._hull

        # keep the hull strictly convex: drop the last vertex while it is not below the new chord
        while len(hull) >= 2:
            u1, v1, _ = hull[-2]
            u2, v2, _ = hull[-1]
            if (v2 - v1) * (u - u2) >= (v - v2) * (u2 - u1):
                hull.pop()
            else:
                break
        hull.append((u, v, n))
        # vertices left of the crossing of the minimum slope serve only changes weaker than mu_min
        slope_min = self._slope_min
        while len(hull) >= 2 and hull[1][1] - hull[0][1] <= slope_min * (hull[1][0] - hull[0][0]):
            hull.popleft()

        best_llr, best_len = 0.0, 0
        for hu, hv, hn in islice(hull, len(hull) - 1):
            llr = self._segment_llr(u - hu, v - hv)
            if llr > best_llr:
                best_llr, best_len = llr, n - hn
        return best_llr, best_len


class GaussianFocus(_HullFocus):
    """
    Online detector of an upward shift in the mean of standardised residuals.

    The input is assumed to be N(0, 1) under H0, e.g. ``(y - y_pred) / y_std``. To detect
    downward shifts, feed ``-x``.

    Parameters
    ----------
    - mu_min (float): Minimum shift, in units of the noise standard deviation, that the test
      looks for. 0 gives standard FOCuS.

    Raises
    ------
    - ValueError: If ``mu_min`` is negative or not finite.

    Examples
    --------
    >>> det = GaussianFocus()
    >>> for x in [0.1, -0.3, 2.0, 2.5]:
    ...     llr, length = det.update(x)
    >>> round((2 * llr) ** 0.5, 3), length
    (3.182, 2)
    """

    def __init__(self, mu_min: float = 0.0) -> None:
        mu_min = float(mu_min)
        if not (isfinite(mu_min) and mu_min >= 0.0):
            raise ValueError(f"mu_min must be a finite number >= 0, got {mu_min}.")
        super().__init__(mu_min, mu_min / 2.0)

    def _segment_llr(self, du: float, dv: float) -> float:
        if dv <= 0.0:
            return 0.0
        mu_min = self.mu_min
        if dv >= mu_min * du:
            return dv * dv / (2.0 * du)
        return max(mu_min * dv - du * mu_min * mu_min / 2.0, 0.0)

    def update(self, x: float) -> tuple[float, int]:
        """
        Process one standardised residual.

        Parameters
        ----------
        - x (float): Residual, N(0, 1) under the null hypothesis.

        Returns
        -------
        - tuple[float, int]: LLR of the most likely change ending at this sample, and the number
          of samples in that change (0 when the LLR is 0).
        """
        return self._push(1.0, float(x))


class PoissonFocus(_HullFocus):
    """
    Online detector of an upward change in the rate of Poisson counts.

    Under H0 the count in each bin is Poisson with a known expectation (the background); after
    the change the expectation is multiplied by an unknown factor ``mu >= mu_min``.

    Parameters
    ----------
    - mu_min (float): Minimum rate multiplier that the test looks for. 1 gives standard
      Poisson-FOCuS.

    Raises
    ------
    - ValueError: If ``mu_min`` is smaller than 1 or not finite.
    """

    def __init__(self, mu_min: float = 1.0) -> None:
        mu_min = float(mu_min)
        if not (isfinite(mu_min) and mu_min >= 1.0):
            raise ValueError(f"mu_min must be a finite number >= 1, got {mu_min}.")
        slope_min = (mu_min - 1.0) / log(mu_min) - 1.0 if mu_min > 1.0 else 0.0
        super().__init__(mu_min, slope_min)

    def _segment_llr(self, du: float, dv: float) -> float:
        if dv <= 0.0:
            return 0.0
        k = du + dv
        mu = max(k / du, self.mu_min)
        return max(k * log(mu) - du * (mu - 1.0), 0.0)

    def update(self, count: float, expected: float) -> tuple[float, int]:
        """
        Process one bin.

        Parameters
        ----------
        - count (float): Observed counts in the bin.
        - expected (float): Background expectation for the bin, strictly positive.

        Returns
        -------
        - tuple[float, int]: LLR of the most likely change ending at this bin, and the number of
          bins in that change (0 when the LLR is 0).

        Raises
        ------
        - ValueError: If ``expected`` is not strictly positive or ``count`` is negative.
        """
        if not expected > 0.0:
            raise ValueError(f"expected must be strictly positive, got {expected}.")
        if count < 0.0:
            raise ValueError(f"count must be non-negative, got {count}.")
        return self._push(float(expected), float(count) - float(expected))


@dataclass(frozen=True)
class FocusResult:
    """
    Output of a batch FOCuS run, one entry per input sample.

    Attributes
    ----------
    - llr (np.ndarray): Maximised log-likelihood ratio.
    - length (np.ndarray): Number of samples in the most likely change ending at each sample, so
      the change spans ``x[t - length[t] + 1 : t + 1]``; 0 where ``llr`` is 0.
    - sign (np.ndarray): +1 for an upward change, -1 for a downward one, 0 where ``llr`` is 0.
    """

    llr: np.ndarray
    length: np.ndarray
    sign: np.ndarray

    @property
    def significance(self) -> np.ndarray:
        """``sqrt(2 * llr)``: not N(0, 1) under H0, see the module docstring."""
        return np.sqrt(2.0 * self.llr)


def _as_finite_1d(name: str, values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
    bad = np.flatnonzero(~np.isfinite(array))
    if bad.size:
        raise ValueError(
            f"{name} contains {bad.size} non-finite values (first at index {bad[0]}). "
            "Drop or impute them, and use `resets` to restart the detector after a gap."
        )
    return array


def _as_resets(resets, size: int) -> np.ndarray:
    if resets is None:
        return np.zeros(size, dtype=bool)
    mask = np.asarray(resets, dtype=bool)
    if mask.shape != (size,):
        raise ValueError(f"resets must have shape ({size},), got {mask.shape}.")
    return mask


def _alarm_llrs(restart_above, background_above) -> tuple[float, float]:
    """LLRs above which the detector restarts, or stores the sample as background (inf: never)."""
    if restart_above is not None and background_above is not None:
        raise ValueError("give either restart_above or background_above, not both.")
    llrs = []
    for name, level in (("restart_above", restart_above), ("background_above", background_above)):
        if level is None:
            llrs.append(np.inf)
            continue
        level = float(level)
        if not (isfinite(level) and level > 0):
            raise ValueError(f"{name} must be a finite positive significance, got {level}.")
        llrs.append(level ** 2 / 2.0)
    return llrs[0], llrs[1]


def _run(detector: _HullFocus, du, dv, resets: np.ndarray, restart_llr: float = np.inf,
         background_llr: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
    llr = np.zeros(len(resets))
    length = np.zeros(len(resets), dtype=np.int64)
    push = detector._push
    keep_state = background_llr < np.inf
    for i, (a, b, reset) in enumerate(zip(du, dv, resets.tolist())):
        if reset:
            detector.reset()
        if keep_state:
            before = detector._state()
        llr[i], length[i] = push(a, b)
        if llr[i] > restart_llr:  # an alarm: start again from scratch
            detector.reset()
        elif llr[i] > background_llr:  # an alarm: remember this sample as background instead
            detector._restore(before)
            push(a, 0.0)
    return llr, length


def _by_side(x: np.ndarray, side: str, one_side) -> FocusResult:
    """Run the one-sided detector ``one_side(values, sign)`` in the requested direction(s)."""
    if side == "up":
        return one_side(x, 1)
    if side == "down":
        return one_side(-x, -1)
    up, down = one_side(x, 1), one_side(-x, -1)
    take_down = down.llr > up.llr
    return FocusResult(
        np.where(take_down, down.llr, up.llr),
        np.where(take_down, down.length, up.length),
        np.where(take_down, down.sign, up.sign),
    )


def gaussian_focus(x, resets=None, side: str = "up", mu_min: float = 0.0,
                   restart_above=None, background_above=None) -> FocusResult:
    """
    Run Gaussian FOCuS over a whole series of standardised residuals.

    Parameters
    ----------
    - x (array-like): Residuals, N(0, 1) under H0, e.g. ``(y - y_pred) / y_std``.
    - resets (Optional[array-like of bool]): ``True`` where the detector must restart before
      processing the sample, e.g. after a data gap.
    - side (str): ``'up'``, ``'down'`` or ``'both'``. With ``'both'`` the two one-sided
      statistics are combined by taking their maximum, which roughly doubles the false-alarm
      rate at a given threshold.
    - mu_min (float): Minimum shift, in noise standard deviations, that the test looks for.
    - restart_above (Optional[float]): If given, the detector starts again from scratch right
      after every sample whose significance exceeds this value (see the module docstring).
    - background_above (Optional[float]): If given, every sample whose significance exceeds this
      value is kept in memory as background (``x = 0``) instead of its value. An alternative to
      ``restart_above`` (see the module docstring).

    Returns
    -------
    - FocusResult: LLR, change length and sign for every sample.

    Raises
    ------
    - ValueError: If ``x`` is not a finite 1-D array, ``resets`` has the wrong shape, ``side``
      is unknown, ``mu_min``, ``restart_above`` or ``background_above`` is invalid, or both of
      the last two are given.

    Examples
    --------
    >>> res = gaussian_focus([0.0, 3.0, 0.0])
    >>> res.significance.round(3).tolist(), res.length.tolist()
    ([0.0, 3.0, 2.121], [0, 1, 2])
    >>> gaussian_focus([1.0, 3.0, 1.0], restart_above=2.5).significance.round(3).tolist()
    [1.0, 3.0, 1.0]
    >>> gaussian_focus([1.0, 3.0, 1.0], background_above=2.5).significance.round(3).tolist()
    [1.0, 3.0, 1.155]
    """
    if side not in ("up", "down", "both"):
        raise ValueError(f"side must be 'up', 'down' or 'both', got {side!r}.")
    GaussianFocus(mu_min)  # validate before touching the data
    restart_llr, background_llr = _alarm_llrs(restart_above, background_above)
    x = _as_finite_1d("x", x)
    resets = _as_resets(resets, len(x))
    ones = [1.0] * len(x)

    def one_side(values: np.ndarray, sign: int) -> FocusResult:
        llr, length = _run(GaussianFocus(mu_min), ones, values.tolist(), resets, restart_llr, background_llr)
        return FocusResult(llr, length, np.where(llr > 0, sign, 0))

    return _by_side(x, side, one_side)


def poisson_focus(counts, expected, resets=None, mu_min: float = 1.0,
                  require_integer_counts: bool = True, restart_above=None,
                  background_above=None) -> FocusResult:
    """
    Run Poisson FOCuS over a whole series of binned counts.

    Parameters
    ----------
    - counts (array-like): Observed counts per bin.
    - expected (array-like): Background expectation per bin, in counts, strictly positive.
    - resets (Optional[array-like of bool]): ``True`` where the detector must restart before
      processing the bin, e.g. after a data gap.
    - mu_min (float): Minimum rate multiplier that the test looks for.
    - require_integer_counts (bool): Reject non-integer ``counts``. Rates (counts per second or
      per unit area) are not counts: multiply them by the exposure first, otherwise the Poisson
      likelihood, and so the significance, is wrong.
    - restart_above (Optional[float]): If given, the detector starts again from scratch right
      after every bin whose significance exceeds this value (see the module docstring).
    - background_above (Optional[float]): If given, every bin whose significance exceeds this
      value is kept in memory as background (as many counts as expected) instead of its counts. An
      alternative to ``restart_above`` (see the module docstring).

    Returns
    -------
    - FocusResult: LLR, change length and sign (+1 or 0) for every bin.

    Raises
    ------
    - ValueError: On non-finite, negative or (if required) non-integer counts, on non-positive
      expectations, on inputs of different lengths, on an invalid ``mu_min``, ``restart_above``
      or ``background_above``, or if both of the last two are given.
    """
    PoissonFocus(mu_min)  # validate before touching the data
    restart_llr, background_llr = _alarm_llrs(restart_above, background_above)
    counts = _as_finite_1d("counts", counts)
    expected = _as_finite_1d("expected", expected)
    if counts.shape != expected.shape:
        raise ValueError(f"counts and expected differ in length: {counts.shape} vs {expected.shape}.")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative.")
    if require_integer_counts and np.any(counts != np.round(counts)):
        raise ValueError(
            "counts are not integers: Poisson-FOCuS needs counts, not rates. Multiply rates by the "
            "exposure, or pass require_integer_counts=False if the values are genuine counts."
        )
    if np.any(expected <= 0):
        raise ValueError("expected must be strictly positive.")
    resets = _as_resets(resets, len(counts))
    llr, length = _run(PoissonFocus(mu_min), expected.tolist(), (counts - expected).tolist(), resets,
                       restart_llr, background_llr)
    return FocusResult(llr, length, np.where(llr > 0, 1, 0))
