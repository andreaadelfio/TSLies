"""
Synthetic data with a known background: every channel is a constant background plus independent
Gaussian noise, one sample per second.

With :func:`~tslies.benchmark.inject` they give fully controlled tests of a trigger, where every
number can be compared with what it should be: the residuals ``(y - background) / std`` are
exactly standard Gaussian noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["gaussian_dataset"]


def gaussian_dataset(n_channels: int, days: float, seed: int = 0, start: str = "2024-01-01",
                     background: float = 100.0, noise_std: float = 10.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Channels ``ch1``...``chN`` with ``y = background + noise_std * z``, where ``z ~ N(0, 1)`` is
    independent from one second and one channel to the next.

    Parameters
    ----------
    - n_channels (int): Number of channels.
    - days (float): Length of the data, one sample per second.
    - seed (int): Seed of the noise.
    - start (str): Time of the first sample (UTC).
    - background (float): Background of every channel.
    - noise_std (float): Standard deviation of the noise.

    Returns
    -------
    - tuple[pd.DataFrame, list[str]]: The data, with ``datetime``, the channels and, for each
      channel, its true background ``<ch>_bkg`` and noise standard deviation ``<ch>_std``; and the
      names of the channels.

    Examples
    --------
    >>> df, cols = gaussian_dataset(2, 3 / 86_400)
    >>> cols, len(df), df["ch1_bkg"].iat[0], df["ch1_std"].iat[0]
    (['ch1', 'ch2'], 3, 100.0, 10.0)
    """
    n = int(round(days * 86_400))
    rng = np.random.default_rng(seed)
    cols = [f"ch{i + 1}" for i in range(n_channels)]
    data = {"datetime": pd.date_range(start, periods=n, freq="s", tz="UTC")}
    for col in cols:
        data[col] = background + noise_std * rng.standard_normal(n)
    for col in cols:
        data[f"{col}_bkg"] = np.full(n, background)
        data[f"{col}_std"] = np.full(n, noise_std)
    return pd.DataFrame(data), cols
