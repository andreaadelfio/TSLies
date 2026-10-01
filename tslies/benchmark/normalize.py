"""
Normalisation of real data with a rolling mean and standard deviation.

``z = (y - rolling_mean) / rolling_std`` turns each channel into residuals that, if the window is
chosen well, behave almost like independent N(0, 1) noise. That is a controlled setting to check
that a trigger statistic behaves as expected, independently of any background model: anomalies
of known strength can be added to ``z`` (or, equivalently, to ``y`` in units of the rolling
standard deviation).

The window is centred and the statistics never mix different data segments. Its length sets what
survives the normalisation: anything slower than about ``window / 2`` samples, whether a slow
fluctuation of the background or a long anomaly, is absorbed by the rolling mean.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .segments import _checked_values, segment_bounds

__all__ = ["rolling_normalize"]


def rolling_normalize(df: pd.DataFrame, y_cols: Sequence[str], window: int, resets=None,
                      min_periods: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling mean and standard deviation of each channel over a centred window, per data segment.

    Parameters
    ----------
    - df (pd.DataFrame): Data, one row per sample.
    - y_cols (Sequence[str]): Channels to normalise.
    - window (int): Window length in samples.
    - resets (Optional[array-like of bool]): ``True`` on the first sample after a data gap; the
      windows never cross a reset.
    - min_periods (Optional[int]): Minimum number of samples in the window to give a value.
      Defaults to ``window // 2``, so that every sample of a segment at least that long gets
      one; shorter segments get NaN, which :class:`tslies.trigger.Trigger` treats as gaps.

    Returns
    -------
    - tuple[pd.DataFrame, pd.DataFrame]: Rolling mean and rolling standard deviation, with the
      columns ``y_cols`` and the index of ``df``. A standard deviation of 0 (flat data) is
      returned as NaN.

    Raises
    ------
    - ValueError: On missing or non-finite columns, ``window < 2`` or an invalid ``min_periods``.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> mean, std = rolling_normalize(df, ['a'], window=3)
    >>> mean['a'].tolist()
    [1.5, 2.0, 3.0, 4.0, 4.5]
    """
    if int(window) != window or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window}.")
    window = int(window)
    min_periods = window // 2 if min_periods is None else int(min_periods)
    if not 1 <= min_periods <= window:
        raise ValueError(f"min_periods must be between 1 and window, got {min_periods}.")
    values = _checked_values(df, y_cols)
    mean = np.full_like(values, np.nan)
    std = np.full_like(values, np.nan)
    for a, b in segment_bounds(len(values), resets):
        rolling = pd.DataFrame(values[a:b]).rolling(window, center=True, min_periods=min_periods)
        mean[a:b] = rolling.mean().to_numpy()
        std[a:b] = rolling.std().to_numpy()
    std[std <= 0] = np.nan
    return (pd.DataFrame(mean, columns=list(y_cols), index=df.index),
            pd.DataFrame(std, columns=list(y_cols), index=df.index))
