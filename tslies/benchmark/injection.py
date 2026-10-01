"""
Injection of synthetic anomalies with known ground truth.

Amplitudes are in units of the local noise standard deviation of each channel (e.g. the
``<channel>_std`` columns of :func:`tslies.benchmark.gaussian_dataset`), so that how hard an
anomaly is to detect does not depend on the scale of the channel.

Shapes (``kind``), all with peak 1 before scaling:

- ``'spike'``: a single sample.
- ``'box'``: a constant level for ``duration`` samples.
- ``'fred'``: fast rise and exponential decay, the pulse of Norris et al. (2005) used for gamma-ray
  burst light curves. It peaks at ``rise_fraction * duration`` and has decayed to 1% of the peak at
  the end.
- ``'gaussian'``: a smooth bump with standard deviation ``duration / 6``: a slow, faint transient.
- ``'triangle'``: a linear rise to the peak at mid-duration and a linear decay: a drift that
  returns to the background.

A negative amplitude gives a dip.

For each anomaly the truth table reports two signal-to-noise ratios, computed for white noise:

- ``z_peak_channel``: ``|A| * max(w) * sum(p) / sqrt(n)``, the z-score of the mean excess over the
  anomaly in its strongest channel, i.e. the significance a single-channel detector that knew the
  anomaly interval would reach.
- ``snr_matched``: ``|A| * sqrt(sum(w**2)) * sqrt(sum(p**2))``, the optimal matched-filter SNR
  combining all channels.

References
----------
- J. P. Norris, J. T. Bonnell, D. Kazanas, J. D. Scargle, J. Hakkila, T. W. Giblin, "Long-lag,
  wide-pulse gamma-ray bursts", ApJ 627 (2005) 324.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .segments import segment_bounds

__all__ = ["KINDS", "DEFAULT_DURATIONS", "Anomaly", "profile", "inject", "random_anomalies"]

KINDS = ("spike", "box", "fred", "gaussian", "triangle")

DEFAULT_DURATIONS = {
    "spike": (1, 1),
    "box": (5, 300),
    "fred": (10, 600),
    "gaussian": (60, 1800),
    "triangle": (60, 1800),
}


def profile(kind: str, duration: int, rise_fraction: float = 0.2) -> np.ndarray:
    """
    Shape of an anomaly, sampled on ``duration`` samples, with maximum 1.

    Parameters
    ----------
    - kind (str): One of :data:`KINDS`.
    - duration (int): Number of samples.
    - rise_fraction (float): Position of the peak of a ``'fred'`` as a fraction of ``duration``.

    Returns
    -------
    - np.ndarray: Array of length ``duration``.

    Raises
    ------
    - ValueError: On an unknown kind, a non-positive duration, a ``'spike'`` longer than one
      sample or ``rise_fraction`` outside (0, 1).

    Examples
    --------
    >>> profile('triangle', 5).round(3).tolist()
    [0.333, 0.667, 1.0, 0.667, 0.333]
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")
    if int(duration) != duration or duration < 1:
        raise ValueError(f"duration must be a positive integer, got {duration}.")
    duration = int(duration)
    if kind == "spike":
        if duration != 1:
            raise ValueError("a spike lasts one sample.")
        return np.ones(1)
    t = np.arange(duration, dtype=float)
    if kind == "box":
        return np.ones(duration)
    if kind == "gaussian":
        shape = np.exp(-0.5 * ((t - (duration - 1) / 2) / (duration / 6)) ** 2)
        return shape / shape.max()
    if kind == "triangle":
        centre = (duration - 1) / 2
        shape = 1 - np.abs(t - centre) / (centre + 1)
        return shape / shape.max()
    if not 0 < rise_fraction < 1:
        raise ValueError("rise_fraction must be in (0, 1).")
    # Norris pulse exp(2 sqrt(t1/t2)) exp(-t1/t - t/t2): peak at sqrt(t1 t2), 1% of it at t = end
    t = t + 0.5
    end = float(duration)
    peak = rise_fraction * end
    tau2 = (end - peak) ** 2 / (end * np.log(100.0))
    tau1 = peak ** 2 / tau2
    shape = np.exp(2 * np.sqrt(tau1 / tau2) - tau1 / t - t / tau2)
    return shape / shape.max()


@dataclass(frozen=True)
class Anomaly:
    """
    One anomaly to inject.

    Attributes
    ----------
    - kind (str): One of :data:`KINDS`.
    - start (int): Position of the first sample (0-based row position, not index label).
    - duration (int): Number of samples.
    - amplitude (float): Peak amplitude in units of the local noise standard deviation; negative
      for a dip.
    - channels (tuple[str, ...]): Affected channels.
    - weights (tuple[float, ...]): Relative amplitude per channel; empty means 1 for all.
    - rise_fraction (float): Peak position of a ``'fred'``.
    """

    kind: str
    start: int
    duration: int
    amplitude: float
    channels: tuple
    weights: tuple = ()
    rise_fraction: float = 0.2

    @property
    def stop(self) -> int:
        """Position of the last sample, inclusive."""
        return self.start + self.duration - 1

    def channel_weights(self) -> np.ndarray:
        """Weights as an array, 1 for every channel when none were given."""
        if not self.weights:
            return np.ones(len(self.channels))
        if len(self.weights) != len(self.channels):
            raise ValueError("weights must have one value per channel.")
        return np.asarray(self.weights, dtype=float)


def inject(df: pd.DataFrame, anomalies: Sequence[Anomaly], std_cols: Optional[Mapping[str, str]] = None,
           std_suffix: str = "_std", time_col: Optional[str] = "datetime"
           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add anomalies to a copy of ``df`` and describe them in a truth table.

    Parameters
    ----------
    - df (pd.DataFrame): Data, one row per sample.
    - anomalies (Sequence[Anomaly]): Anomalies to add; overlapping ones add up.
    - std_cols (Optional[Mapping[str, str]]): Column with the noise standard deviation of each
      channel. Defaults to ``<channel><std_suffix>``.
    - std_suffix (str): Suffix of the default standard deviation columns.
    - time_col (Optional[str]): Timestamp column reported in the truth table, if any.

    Returns
    -------
    - tuple[pd.DataFrame, pd.DataFrame]: The data with the anomalies, plus ``<channel>_injected``
      (the added signal) for every affected channel and ``injected_id`` (id of the anomaly at each
      row, -1 elsewhere); and the truth table, one row per anomaly.

    Raises
    ------
    - ValueError: On anomalies outside the data, unknown channels or invalid shapes.
    """
    out = df.copy()
    n = len(df)
    added: dict[str, np.ndarray] = {}
    injected_id = np.full(n, -1, dtype=np.int64)
    rows = []
    for anomaly_id, anomaly in enumerate(anomalies):
        if anomaly.start < 0 or anomaly.stop >= n:
            raise ValueError(f"anomaly {anomaly_id} spans rows {anomaly.start}-{anomaly.stop}, outside the data.")
        shape = profile(anomaly.kind, anomaly.duration, anomaly.rise_fraction)
        weights = anomaly.channel_weights()
        span = slice(anomaly.start, anomaly.stop + 1)
        for channel, weight in zip(anomaly.channels, weights):
            std_col = (std_cols or {}).get(channel, f"{channel}{std_suffix}")
            for col in (channel, std_col):
                if col not in df.columns:
                    raise ValueError(f"anomaly {anomaly_id}: df has no column {col!r}.")
            signal = added.setdefault(channel, np.zeros(n))
            signal[span] += anomaly.amplitude * weight * df[std_col].to_numpy(dtype=float)[span] * shape
        injected_id[span] = anomaly_id

        peak = anomaly.start + int(np.argmax(shape))
        row = {
            "anomaly_id": anomaly_id,
            "kind": anomaly.kind,
            "amplitude": anomaly.amplitude,
            "duration": anomaly.duration,
            "start_index": anomaly.start,
            "peak_index": peak,
            "stop_index": anomaly.stop,
            "channels": "/".join(anomaly.channels),
            "weights": "/".join(f"{w:.3g}" for w in weights),
            "n_channels": len(anomaly.channels),
            "rise_fraction": anomaly.rise_fraction if anomaly.kind == "fred" else np.nan,
            "z_peak_channel": abs(anomaly.amplitude) * np.max(np.abs(weights)) * shape.sum() / np.sqrt(anomaly.duration),
            "snr_matched": abs(anomaly.amplitude) * np.sqrt(np.sum(weights ** 2) * np.sum(shape ** 2)),
        }
        if time_col is not None and time_col in df.columns:
            times = df[time_col]
            row.update({f"start_{time_col}": times.iat[anomaly.start], f"peak_{time_col}": times.iat[peak],
                        f"stop_{time_col}": times.iat[anomaly.stop]})
        rows.append(row)

    for channel, signal in added.items():
        out[channel] = out[channel].to_numpy(dtype=float) + signal
        out[f"{channel}_injected"] = signal
    out["injected_id"] = injected_id
    return out, pd.DataFrame(rows)


def random_anomalies(n_anomalies: int, n_samples: int, channels: Sequence[str], resets=None,
                     groups: Optional[Mapping[str, Sequence[str]]] = None,
                     kinds: Sequence[str] = KINDS,
                     z_range: tuple[float, float] = (3.0, 30.0),
                     durations: Optional[Mapping[str, tuple[int, int]]] = None,
                     n_groups: tuple[int, int] = (1, 1),
                     weight_range: tuple[float, float] = (0.5, 1.0),
                     negative_fraction: float = 0.0,
                     rise_fraction_range: tuple[float, float] = (0.05, 0.3),
                     margin: int = 100, min_gap: int = 300,
                     seed: Optional[int] = None, max_tries: int = 10_000) -> list[Anomaly]:
    """
    Draw non-overlapping anomalies of random kind, duration, strength, position and channels.

    The strength is drawn log-uniformly as ``z_peak_channel`` in ``z_range`` (see the module
    docstring), and the amplitude follows from it, so that short and long anomalies cover the same
    range of detectability.

    Parameters
    ----------
    - n_anomalies (int): Number of anomalies.
    - n_samples (int): Length of the data.
    - channels (Sequence[str]): Channels that can be affected.
    - resets (Optional[array-like of bool]): Segment starts; an anomaly never spans a reset.
    - groups (Optional[Mapping[str, Sequence[str]]]): Channels that are affected together, e.g. the
      energy bands of a detector face. Defaults to one group per channel.
    - kinds (Sequence[str]): Kinds to draw from, uniformly.
    - z_range (tuple[float, float]): Range of ``z_peak_channel``.
    - durations (Optional[Mapping[str, tuple[int, int]]]): Duration range per kind, drawn
      log-uniformly. Defaults to :data:`DEFAULT_DURATIONS`.
    - n_groups (tuple[int, int]): Range of the number of affected groups.
    - weight_range (tuple[float, float]): Range of the channel weights; the strongest channel of
      each anomaly has weight 1.
    - negative_fraction (float): Probability that an anomaly is a dip.
    - rise_fraction_range (tuple[float, float]): Range of the peak position of a ``'fred'``.
    - margin (int): Minimum distance, in samples, from the edges of a segment.
    - min_gap (int): Minimum distance, in samples, between two anomalies.
    - seed (Optional[int]): Random seed.
    - max_tries (int): Placement attempts before giving up.

    Returns
    -------
    - list[Anomaly]: Anomalies sorted by start.

    Raises
    ------
    - ValueError: On invalid options, or if the anomalies do not fit in the data.
    """
    rng = np.random.default_rng(seed)
    durations = {**DEFAULT_DURATIONS, **(durations or {})}
    groups = {c: [c] for c in channels} if groups is None else {g: list(m) for g, m in groups.items()}
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown kinds {unknown}.")
    if not 0 < z_range[0] <= z_range[1]:
        raise ValueError("z_range must be positive and increasing.")
    if not 1 <= n_groups[0] <= n_groups[1] <= len(groups):
        raise ValueError(f"n_groups must be within [1, {len(groups)}].")
    if not 0 < weight_range[0] <= weight_range[1] <= 1:
        raise ValueError("weight_range must lie in (0, 1].")
    if not 0 <= negative_fraction <= 1:
        raise ValueError("negative_fraction must be a probability.")

    bounds = segment_bounds(n_samples, resets)
    placed: list[tuple[int, int]] = []
    anomalies = []
    tries = 0
    while len(anomalies) < n_anomalies:
        tries += 1
        if tries > max_tries:
            raise ValueError(f"could only place {len(anomalies)} of {n_anomalies} anomalies: "
                             "reduce their number, durations, margin or min_gap.")
        kind = kinds[rng.integers(len(kinds))]
        lo, hi = durations[kind]
        duration = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        room = np.array([max(b - a - 2 * margin - duration + 1, 0) for a, b in bounds], dtype=float)
        if room.sum() == 0:
            continue
        seg = rng.choice(len(bounds), p=room / room.sum())
        start = bounds[seg][0] + margin + int(rng.integers(room[seg]))
        stop = start + duration - 1
        if any(start - min_gap <= b and stop + min_gap >= a for a, b in placed):
            continue

        names = rng.choice(list(groups), size=rng.integers(n_groups[0], n_groups[1] + 1), replace=False)
        affected = [c for g in names for c in groups[g]]
        weights = rng.uniform(weight_range[0], weight_range[1], len(affected))
        weights /= weights.max()
        rise = float(rng.uniform(*rise_fraction_range))
        shape = profile(kind, duration, rise)
        z = np.exp(rng.uniform(np.log(z_range[0]), np.log(z_range[1])))
        amplitude = z * np.sqrt(duration) / shape.sum()
        if rng.uniform() < negative_fraction:
            amplitude = -amplitude
        anomalies.append(Anomaly(kind, start, duration, float(amplitude), tuple(affected),
                                 tuple(float(w) for w in weights), rise))
        placed.append((start, stop))
    return sorted(anomalies, key=lambda a: a.start)
