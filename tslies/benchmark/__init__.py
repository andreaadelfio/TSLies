"""
Benchmark data for anomaly detection: synthetic Gaussian data with a known background, or real data
normalised with a rolling mean and standard deviation, with injected anomalies of known strength.

Typical use::

    mean, std = rolling_normalize(real_df, y_cols, window=600, resets=resets)
    df = real_df.join(mean.add_suffix("_mean")).join(std.add_suffix("_std"))
    anomalies = random_anomalies(50, len(df), y_cols, resets=resets, seed=1)
    data, truth = inject(df, anomalies, std_suffix="_std")  # amplitudes in units of the rolling std
    # ... run a Trigger on `data` with the `_mean` columns as background, then
    truth_out, events_out, summary = match_events(trigger.get_detections_df(), truth)

This subpackage only depends on numpy and pandas and has no side effects at import time.
"""

from .evaluation import match_events
from .injection import DEFAULT_DURATIONS, KINDS, Anomaly, inject, profile, random_anomalies
from .normalize import rolling_normalize
from .segments import segment_bounds
from .synthetic import gaussian_dataset

__all__ = [
    "Anomaly",
    "DEFAULT_DURATIONS",
    "KINDS",
    "gaussian_dataset",
    "inject",
    "match_events",
    "profile",
    "random_anomalies",
    "rolling_normalize",
    "segment_bounds",
]
