"""
Data segments (the rows between two resets, e.g. two data gaps) and input checks shared by the
benchmark tools.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["segment_bounds"]


def segment_bounds(n: int, resets=None) -> list[tuple[int, int]]:
    """
    Split ``range(n)`` into segments that start at every reset.

    Parameters
    ----------
    - n (int): Number of samples.
    - resets (Optional[array-like of bool]): ``True`` on the first sample of a new segment.

    Returns
    -------
    - list[tuple[int, int]]: ``(start, stop)`` of each segment, ``stop`` excluded.

    Raises
    ------
    - ValueError: If ``resets`` does not have length ``n``.
    """
    if resets is None:
        starts = [0]
    else:
        mask = np.asarray(resets, dtype=bool)
        if mask.shape != (n,):
            raise ValueError(f"resets must have shape ({n},), got {mask.shape}.")
        starts = sorted({0, *np.flatnonzero(mask).tolist()})
    stops = starts[1:] + [n]
    return [(a, b) for a, b in zip(starts, stops) if b > a]


def _checked_values(df: pd.DataFrame, y_cols: Sequence[str]) -> np.ndarray:
    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing the columns {missing}.")
    values = df[list(y_cols)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("y_cols contain non-finite values: drop or impute them first.")
    return values
