"""
Comparison of detected events with injected anomalies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["match_events"]


def match_events(detections: pd.DataFrame, truth: pd.DataFrame, tolerance: int = 0,
                 stop_col: str = "stop_index") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Match detected events and injected anomalies by overlap of their sample intervals.

    Parameters
    ----------
    - detections (pd.DataFrame): Events with ``start_index`` and ``stop_index`` (inclusive), and
      optionally ``detection_index`` and ``event_id``, e.g. from
      :meth:`tslies.trigger.Trigger.get_detections_df`.
    - truth (pd.DataFrame): Anomalies with ``anomaly_id``, ``start_index`` and ``stop_index``,
      e.g. from :func:`tslies.benchmark.inject`.
    - tolerance (int): Samples added on both sides of each anomaly when matching.
    - stop_col (str): Column of ``detections`` that ends the interval of an event, e.g.
      ``'stop_index'`` (its last anomalous sample) or ``'peak_index'``.

    Returns
    -------
    - tuple[pd.DataFrame, pd.DataFrame, dict]:

      - ``truth`` with ``detected``, ``event_ids`` and, for the matching event with the largest
        overlap, ``start_error`` (event start minus anomaly start) and ``delay`` (first sample over
        threshold minus anomaly start), in samples;
      - ``detections`` with ``matched`` and ``anomaly_ids``;
      - a summary with ``n_anomalies``, ``n_detected``, ``efficiency``, ``n_events`` and
        ``n_false`` (events matching no anomaly).

    Raises
    ------
    - ValueError: If a required column is missing.
    """
    for name, table, cols in (("detections", detections, ("start_index", stop_col)),
                              ("truth", truth, ("anomaly_id", "start_index", "stop_index"))):
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise ValueError(f"{name} is missing the columns {missing}.")

    det_start = detections["start_index"].to_numpy()
    det_stop = detections[stop_col].to_numpy()
    tru_start = truth["start_index"].to_numpy() - tolerance
    tru_stop = truth["stop_index"].to_numpy() + tolerance
    overlap = (det_start[:, None] <= tru_stop[None, :]) & (det_stop[:, None] >= tru_start[None, :])

    event_ids = detections["event_id"].to_numpy() if "event_id" in detections else np.arange(len(detections))
    first_sample = (detections["detection_index"] if "detection_index" in detections
                    else detections["start_index"]).to_numpy()

    truth_out = truth.copy()
    truth_out["detected"] = overlap.any(axis=0)
    truth_out["event_ids"] = [event_ids[overlap[:, j]].tolist() for j in range(len(truth))]
    start_error = np.full(len(truth), np.nan)
    delay = np.full(len(truth), np.nan)
    shared = (np.minimum(det_stop[:, None], tru_stop[None, :])
              - np.maximum(det_start[:, None], tru_start[None, :]) + 1)
    for j in np.flatnonzero(truth_out["detected"].to_numpy()):
        i = int(np.argmax(np.where(overlap[:, j], shared[:, j], -1)))
        start_error[j] = det_start[i] - truth["start_index"].iat[j]
        delay[j] = first_sample[i] - truth["start_index"].iat[j]
    truth_out["start_error"] = start_error
    truth_out["delay"] = delay

    anomaly_ids = truth["anomaly_id"].to_numpy()
    det_out = detections.copy()
    det_out["matched"] = overlap.any(axis=1)
    det_out["anomaly_ids"] = [anomaly_ids[overlap[i]].tolist() for i in range(len(detections))]

    n_detected = int(truth_out["detected"].sum())
    summary = {
        "n_anomalies": len(truth),
        "n_detected": n_detected,
        "efficiency": n_detected / len(truth) if len(truth) else np.nan,
        "n_events": len(detections),
        "n_false": int((~det_out["matched"]).sum()),
    }
    return truth_out, det_out, summary
