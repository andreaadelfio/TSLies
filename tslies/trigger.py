"""
Anomaly triggers on the residuals of a background model.

:class:`Trigger` turns observed signals and their predicted background into a significance per
sample and channel, applies per-channel thresholds and a coincidence condition across groups of
channels, and groups the triggered samples into events.

Detectors (``trigger_type``)
----------------------------
- ``'z_score'``: evidence of each single sample, from ``z = (y - y_pred) / y_std``.
- ``'focus'`` (alias ``'gaussian_focus'``): Gaussian FOCuS on ``z``, which must be N(0, 1) when
  there is no anomaly.
- ``'poisson_focus'``: Poisson-FOCuS on counts ``y`` with expected counts ``y_pred``. It needs no
  ``y_std``, but ``y`` must be counts, not rates.

The significance of the FOCuS detectors is ``sqrt(2 * LLR)``. Because it is maximised over the
start of the change it is not N(0, 1) without anomalies: thresholds must be calibrated for a
target false-alarm rate (see :mod:`tslies.stats.focus`).

An **alarm** is a sample where the significance of a channel goes over its threshold. By default
(``after_alarm='restart'``) the FOCuS detectors restart from scratch after every alarm, as FOCuS
is meant to be used on a stream: each alarm reports one change, from its estimated start to the
alarm, which is then forgotten, and a long anomaly raises a sequence of alarms, one after the
other, while it lasts. Without restarts the evidence of a strong anomaly would keep the
significance over threshold for hours after its end and hide the anomalies that follow. With
``after_alarm='background'`` they keep their memory instead, with the sample that raised the
alarm stored as background: a long anomaly raises an alarm at almost every sample, but alarms can
keep coming after its end and anomalies hours apart can end up in the same event (see
:mod:`tslies.stats.focus`). Restarting is the safer choice.

Provenance
----------
This module replaces an earlier implementation adapted from the FOCuS reference notebooks of
K. Ward (https://github.com/kesward/FOCuS) and from the Poisson-FOCuS trigger of DeepGRB
(https://github.com/rcrupi/DeepGRB; R. Crupi et al., Experimental Astronomy 2023,
doi:10.1007/s10686-023-09915-7). The detectors now live in :mod:`tslies.stats.focus`. The
coincidence condition across groups of channels follows the DeepGRB trigger, which requires at
least two detectors over threshold at the same time.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .stats.focus import gaussian_focus, poisson_focus

logger = logging.getLogger(__name__)

_TRIGGER_TYPES = {
    "z_score": "z_score",
    "focus": "focus",
    "gauss_focus": "focus",
    "gaussian_focus": "focus",
    "poisson_focus": "poisson_focus",
}


class Trigger:
    """
    Detect anomalies in multichannel time series from observed values and predicted background.

    Parameters
    ----------
    - tiles_df (pd.DataFrame): One row per time bin, holding observed, predicted and (for the
      Gaussian detectors) standard-deviation columns, plus ``time_col``. It is never modified.
    - y_cols (Sequence[str]): Observed channels.
    - y_cols_pred (Sequence[str]): Predicted background, one column per channel of ``y_cols``.
    - thresholds (Optional[float | Mapping[str, float]]): Significance threshold, one for all
      channels or one per channel. ``None`` means 5 for every channel.
    - trigger_type (str): ``'focus'``, ``'poisson_focus'`` or ``'z_score'``.
    - units (Optional[Mapping[str, str]]): Units per column, used by the plots.
    - latex_y_cols (Optional[Mapping[str, str]]): LaTeX labels per channel, used by the plots.
    - std_cols (Optional[Sequence[str]]): Standard deviation of the background prediction, one per
      channel. Defaults to ``'<channel>_std'``. Not used by ``'poisson_focus'``.
    - side (str): ``'up'``, ``'down'`` or ``'both'``: direction of the anomalies to look for.
      ``'poisson_focus'`` only supports ``'up'``.
    - mu_min (Optional[float]): Minimum intensity of the change looked for by the FOCuS detectors:
      a shift in noise standard deviations (``'focus'``, default 0) or a rate multiplier
      (``'poisson_focus'``, default 1). See :mod:`tslies.stats.focus`.
    - after_alarm (str): What the FOCuS detectors do after every alarm: ``'restart'`` from
      scratch (the default) or keep their memory with the alarmed sample stored as
      ``'background'``. See the module docstring.
    - groups (Optional[Mapping[str, Sequence[str]]]): Partition of ``y_cols`` into groups, e.g.
      the energy bands of one detector face. A group is triggered when any of its channels is.
      Defaults to one group per channel.
    - min_groups (int): Minimum number of groups triggered at the same time for a sample to be
      anomalous. 1 is a logical OR over groups.
    - time_col (str): Column with the timestamp of each row.

    Raises
    ------
    - ValueError: On inconsistent columns, thresholds, groups or options.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'datetime': pd.date_range('2024-01-01', periods=300, freq='s'),
    ...                    'a': rng.normal(size=300), 'a_pred': 0.0, 'a_std': 1.0})
    >>> df.loc[100:129, 'a'] += 2.0
    >>> trig = Trigger(df, ['a'], ['a_pred'], thresholds=6.0)
    >>> _ = trig.run()
    >>> events, _ = trig.identify_and_merge_triggers()
    >>> len(events)
    1
    """

    def __init__(
        self,
        tiles_df: pd.DataFrame,
        y_cols: Sequence[str],
        y_cols_pred: Sequence[str],
        thresholds: Optional[Union[float, Mapping[str, float]]] = None,
        trigger_type: str = "focus",
        units: Optional[Mapping[str, str]] = None,
        latex_y_cols: Optional[Mapping[str, str]] = None,
        *,
        std_cols: Optional[Sequence[str]] = None,
        side: str = "up",
        mu_min: Optional[float] = None,
        after_alarm: str = "restart",
        groups: Optional[Mapping[str, Sequence[str]]] = None,
        min_groups: int = 1,
        time_col: str = "datetime",
    ):
        if not isinstance(tiles_df, pd.DataFrame):
            raise ValueError("tiles_df must be a pandas DataFrame.")
        self.y_cols = list(y_cols)
        self.y_cols_pred = list(y_cols_pred)
        if not self.y_cols:
            raise ValueError("y_cols must contain at least one channel.")
        if len(set(self.y_cols)) != len(self.y_cols):
            raise ValueError("y_cols contains duplicated channels.")
        if len(self.y_cols_pred) != len(self.y_cols):
            raise ValueError(
                f"y_cols_pred must have one column per channel: got {len(self.y_cols_pred)} "
                f"for {len(self.y_cols)} channels."
            )

        if trigger_type not in _TRIGGER_TYPES:
            raise ValueError(f"trigger_type must be one of {sorted(_TRIGGER_TYPES)}, got {trigger_type!r}.")
        self.trigger_type = trigger_type
        self._detector = _TRIGGER_TYPES[trigger_type]

        if side not in ("up", "down", "both"):
            raise ValueError(f"side must be 'up', 'down' or 'both', got {side!r}.")
        if self._detector == "poisson_focus" and side != "up":
            raise ValueError("poisson_focus only detects increases: use side='up'.")
        self.side = side

        if self._detector == "z_score":
            if mu_min is not None:
                raise ValueError("mu_min only applies to the FOCuS detectors, not to 'z_score'.")
        elif mu_min is None:
            mu_min = 1.0 if self._detector == "poisson_focus" else 0.0
        self.mu_min = mu_min

        if after_alarm not in ("restart", "background"):
            raise ValueError(f"after_alarm must be 'restart' or 'background', got {after_alarm!r}.")
        if self._detector == "z_score" and after_alarm != "restart":
            raise ValueError("after_alarm only applies to the FOCuS detectors, not to 'z_score'.")
        self.after_alarm = after_alarm

        if self._detector == "poisson_focus":
            self.std_cols = None
        else:
            self.std_cols = list(std_cols) if std_cols is not None else [f"{c}_std" for c in self.y_cols]
            if len(self.std_cols) != len(self.y_cols):
                raise ValueError("std_cols must have one column per channel.")

        self.time_col = time_col
        needed = self.y_cols + self.y_cols_pred + (self.std_cols or []) + [time_col]
        missing = [c for c in needed if c not in tiles_df.columns]
        if missing:
            raise ValueError(f"tiles_df is missing the columns {missing}.")
        self.tiles_df = tiles_df

        self.thresholds = self._as_thresholds(thresholds)
        self.groups = self._as_groups(groups)
        if not 1 <= int(min_groups) <= len(self.groups):
            raise ValueError(f"min_groups must be between 1 and the number of groups ({len(self.groups)}).")
        self.min_groups = int(min_groups)

        self.units = dict(units or {})
        self.latex_y_cols = dict(latex_y_cols or {})

        self.results: Optional[pd.DataFrame] = None
        self.return_df: Optional[pd.DataFrame] = None
        self.mask: Optional[np.ndarray] = None
        self.merged_anomalies: dict = {}
        self._resets: Optional[np.ndarray] = None

    def _as_thresholds(self, thresholds) -> dict:
        if thresholds is None:
            thresholds = 5.0
        if isinstance(thresholds, Mapping):
            missing = [c for c in self.y_cols if c not in thresholds]
            if missing:
                raise ValueError(f"thresholds is missing the channels {missing}.")
            values = {c: float(thresholds[c]) for c in self.y_cols}
        else:
            values = {c: float(thresholds) for c in self.y_cols}
        bad = [c for c, v in values.items() if not (np.isfinite(v) and v > 0)]
        if bad:
            raise ValueError(f"thresholds must be finite and positive, invalid for {bad}.")
        return values

    def _as_groups(self, groups) -> dict:
        if groups is None:
            return {c: [c] for c in self.y_cols}
        groups = {name: list(members) for name, members in groups.items()}
        members = [c for cols in groups.values() for c in cols]
        unknown = sorted(set(members) - set(self.y_cols))
        repeated = sorted({c for c in members if members.count(c) > 1})
        uncovered = [c for c in self.y_cols if c not in members]
        if unknown or repeated or uncovered or any(not cols for cols in groups.values()):
            raise ValueError(
                "groups must partition y_cols into non-empty groups: "
                f"unknown {unknown}, repeated {repeated}, not assigned {uncovered}."
            )
        return groups

    def _channel_significance(self, face: str, face_pred: str, std_col: Optional[str],
                              resets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Significance, change length and sign of one channel; NaN where the input is unusable."""
        y = self.tiles_df[face].to_numpy(dtype=float)
        pred = self.tiles_df[face_pred].to_numpy(dtype=float)
        if self._detector == "poisson_focus":
            finite = np.isfinite(y) & np.isfinite(pred) & (pred > 0)
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (y - pred) / self.tiles_df[std_col].to_numpy(dtype=float)
            finite = np.isfinite(z)
        n_bad = int((~finite).sum())
        if n_bad:
            logger.warning("%s: %d samples with non-finite residuals or non-positive background "
                           "are treated as gaps.", face, n_bad)

        significance = np.full(len(y), np.nan)
        length = np.zeros(len(y), dtype=np.int64)
        sign = np.zeros(len(y), dtype=np.int64)
        idx = np.flatnonzero(finite)
        if idx.size == 0:
            return significance, length, sign

        if self._detector == "z_score":
            zs = z[idx]
            evidence = {"up": zs, "down": -zs, "both": np.abs(zs)}[self.side]
            significance[idx] = np.maximum(evidence, 0.0)
            length[idx] = (evidence > 0).astype(np.int64)
            sign[idx] = np.where(evidence > 0, np.sign(zs), 0).astype(np.int64)
            return significance, length, sign

        # restart the detector after a gap, and when a reset falls on the gap itself
        cum_resets = np.cumsum(resets)
        sub_resets = resets[idx].copy()
        sub_resets[1:] |= (np.diff(idx) > 1) | (np.diff(cum_resets[idx]) > 0)
        # what the detector does after every alarm
        threshold = self.thresholds[face]
        after_alarm = {"restart_above": threshold} if self.after_alarm == "restart" else {"background_above": threshold}
        if self._detector == "focus":
            res = gaussian_focus(z[idx], resets=sub_resets, side=self.side, mu_min=self.mu_min, **after_alarm)
        else:
            res = poisson_focus(y[idx], pred[idx], resets=sub_resets, mu_min=self.mu_min, **after_alarm)
        significance[idx] = res.significance
        length[idx] = res.length
        sign[idx] = res.sign
        return significance, length, sign

    def run(self, reset_condition=None) -> pd.DataFrame:
        """
        Compute the significance of every channel and flag the anomalous samples.

        Parameters
        ----------
        - reset_condition (Optional[array-like of bool]): ``True`` on the first sample after a
          data gap (e.g. an SAA passage): the detectors restart there, and events never span it.

        Returns
        -------
        - pd.DataFrame: Copy of ``tiles_df`` with an ``anomaly`` column: 1 where at least
          ``min_groups`` groups are anomalous. A channel is anomalous over the stretch of data that
          raised each of its alarms, from the estimated start of the change to the alarm (for
          ``'z_score'``, the alarm alone). The per-sample details are in ``results``:
          ``<channel>_significance``, ``<channel>_length`` (samples in the change ending at each
          sample) and ``<channel>_triggered`` (anomalous samples).

        Raises
        ------
        - ValueError: If ``reset_condition`` has the wrong length, or on invalid counts for
          ``'poisson_focus'``.
        """
        n = len(self.tiles_df)
        if reset_condition is None:
            resets = np.zeros(n, dtype=bool)
        else:
            resets = np.asarray(reset_condition, dtype=bool)
            if resets.shape != (n,):
                raise ValueError(f"reset_condition must have shape ({n},), got {resets.shape}.")
        self._resets = resets

        columns = {}
        triggered = {}
        std_cols = self.std_cols or [None] * len(self.y_cols)
        for face, face_pred, std_col in zip(self.y_cols, self.y_cols_pred, std_cols):
            significance, length, sign = self._channel_significance(face, face_pred, std_col, resets)
            alarms = np.flatnonzero(np.nan_to_num(significance, nan=0.0) > self.thresholds[face])
            edges = np.zeros(n + 1, dtype=np.int64)  # mark each change, from its start to its alarm
            np.add.at(edges, alarms - length[alarms] + 1, 1)
            np.add.at(edges, alarms + 1, -1)
            triggered[face] = np.cumsum(edges[:-1]) > 0
            columns[f"{face}_significance"] = significance
            columns[f"{face}_length"] = length
            columns[f"{face}_triggered"] = triggered[face]
            if self.side == "both":
                columns[f"{face}_sign"] = sign
        columns[self.time_col] = self.tiles_df[self.time_col].to_numpy()
        self.results = pd.DataFrame(columns)
        self.return_df = self.results

        groups_triggered = np.zeros(n, dtype=np.int64)
        for members in self.groups.values():
            groups_triggered += np.any([triggered[c] for c in members], axis=0)
        self.mask = groups_triggered >= self.min_groups

        out = self.tiles_df.copy()
        out["anomaly"] = self.mask.astype(int)
        return out

    def identify_and_merge_triggers(self, merge_interval: int = 60) -> tuple[dict, pd.DataFrame]:
        """
        Group the anomalous samples into events.

        An event is a run of anomalous samples (see :meth:`run`) of the same data segment; runs
        separated by at most ``merge_interval`` non-anomalous samples are merged. When the FOCuS
        detectors restart after every alarm (the default), a long anomaly raises a sequence of alarms, and the
        changes that raise them follow one another, but with short gaps: after a restart the
        estimated start of the next change can fall some tens of samples after the previous alarm,
        where the noise happened to be low. ``merge_interval`` must be long enough to bridge those
        gaps, or a long anomaly is split into several events.

        For every channel of an event: ``start_index`` is the first anomalous sample (the
        estimated start of its first change), ``detection_index`` its first alarm, ``peak_index``
        its alarm with the highest significance and ``stop_index`` its last anomalous sample.

        Parameters
        ----------
        - merge_interval (int): Maximum number of non-anomalous samples between two runs that are
          still merged into one event: 60 samples is one minute of 1 s bins, as in the TSLies
          examples.

        Returns
        -------
        - tuple[dict, pd.DataFrame]: ``{event_id: {channel: info}}``, where ``info`` holds
          ``start_index``, ``detection_index``, ``peak_index``, ``stop_index`` (positions in
          ``tiles_df``, inclusive), ``max_significance`` and the corresponding timestamps; and the
          per-sample ``results``.

        Raises
        ------
        - RuntimeError: If :meth:`run` has not been called.
        """
        if self.mask is None:
            raise RuntimeError("Call run() before identify_and_merge_triggers().")
        if merge_interval < 0:
            raise ValueError("merge_interval must be non-negative.")
        self.merged_anomalies = {}
        mask = self.mask
        if not mask.any():
            logger.info("No triggers detected.")
            return self.merged_anomalies, self.results

        edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
        run_starts = np.flatnonzero(edges == 1)
        run_stops = np.flatnonzero(edges == -1) - 1
        segment = np.cumsum(self._resets)
        spans = []
        for a, b in zip(run_starts, run_stops):
            if spans and a - spans[-1][1] - 1 <= merge_interval and segment[a] == segment[spans[-1][1]]:
                spans[-1][1] = b
            else:
                spans.append([a, b])

        times = self.results[self.time_col]
        for event_id, (a, b) in enumerate(spans):
            event = {}
            for face in self.y_cols:
                idx = np.flatnonzero(self.results[f"{face}_triggered"].to_numpy()[a:b + 1]) + a
                if idx.size == 0:
                    continue
                sig = np.nan_to_num(self.results[f"{face}_significance"].to_numpy(), nan=0.0)
                alarms = idx[sig[idx] > self.thresholds[face]]
                if alarms.size == 0:  # the alarm fell outside the coincidence with other groups
                    alarms = idx
                peak = int(alarms[np.argmax(sig[alarms])])
                start, detection, stop = int(idx[0]), int(alarms[0]), int(idx[-1])
                event[face] = {
                    "start_index": start,
                    "detection_index": detection,
                    "peak_index": peak,
                    "stop_index": stop,
                    "max_significance": float(sig[peak]),
                    f"start_{self.time_col}": times.iat[start],
                    f"detection_{self.time_col}": times.iat[detection],
                    f"peak_{self.time_col}": times.iat[peak],
                    f"stop_{self.time_col}": times.iat[stop],
                }
            self.merged_anomalies[event_id] = event
        logger.info("%d events from %d runs of anomalous samples.", len(spans), len(run_starts))
        return self.merged_anomalies, self.results

    def get_detections_df(self, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Summarise the events, one row each, in chronological order.

        Parameters
        ----------
        - cols (Optional[Iterable[str]]): Extra columns of ``tiles_df`` to report at the start
          and at the stop of each event, e.g. ``['MET']``.

        Returns
        -------
        - pd.DataFrame: ``event_id``, ``start_<col>`` and ``stop_<col>`` for ``time_col`` and
          ``cols``, the detection (first sample over threshold) and peak times,
          ``triggered_faces`` ('/'-separated), ``n_channels``, ``peak_channel``,
          ``max_significance`` and the start/detection/peak/stop positions.
        """
        keys = [self.time_col] + [c for c in (cols or []) if c != self.time_col]
        rows = []
        for event_id, event in self.merged_anomalies.items():
            start = min(info["start_index"] for info in event.values())
            detection = min(info["detection_index"] for info in event.values())
            stop = max(info["stop_index"] for info in event.values())
            peak_channel = max(event, key=lambda face: event[face]["max_significance"])
            row = {"event_id": event_id}
            for key in keys:
                row[f"start_{key}"] = self.tiles_df[key].iat[start]
                row[f"stop_{key}"] = self.tiles_df[key].iat[stop]
            row[f"detection_{self.time_col}"] = self.tiles_df[self.time_col].iat[detection]
            row[f"peak_{self.time_col}"] = self.tiles_df[self.time_col].iat[event[peak_channel]["peak_index"]]
            row.update({
                "triggered_faces": "/".join(event),
                "n_channels": len(event),
                "peak_channel": peak_channel,
                "max_significance": event[peak_channel]["max_significance"],
                "start_index": start,
                "detection_index": detection,
                "peak_index": event[peak_channel]["peak_index"],
                "stop_index": stop,
            })
            rows.append(row)
        columns = (["event_id"] + [f"{p}_{k}" for k in keys for p in ("start", "stop")]
                   + [f"detection_{self.time_col}", f"peak_{self.time_col}", "triggered_faces",
                      "n_channels", "peak_channel", "max_significance", "start_index",
                      "detection_index", "peak_index", "stop_index"])
        return pd.DataFrame(rows, columns=columns).sort_values("start_index", ignore_index=True)

    def save_detections_csv(self, detections_df: pd.DataFrame, file: str = "", suffix: str = "",
                            folder: Optional[Union[str, os.PathLike]] = None) -> str:
        """
        Write a detections table to ``detections[_file][suffix].csv``.

        Parameters
        ----------
        - detections_df (pd.DataFrame): Table to write, e.g. from :meth:`get_detections_df`.
        - file (str): Optional name inserted after ``detections_``.
        - suffix (str): Optional suffix appended to the file name.
        - folder (Optional[str | os.PathLike]): Destination folder. Defaults to the anomalies
          folder of the current TSLies session (``tslies.config.ANOMALIES_TIME_DIR``).

        Returns
        -------
        - str: Path of the written file.
        """
        if folder is None:
            from .config import ANOMALIES_TIME_DIR
            if ANOMALIES_TIME_DIR is None:
                raise RuntimeError("TSLies base directory is not configured: pass folder explicitly.")
            folder = ANOMALIES_TIME_DIR
        os.makedirs(folder, exist_ok=True)
        name = f"detections{'_' + file if file else ''}{suffix}.csv"
        path = os.path.join(folder, name)
        detections_df.to_csv(path, index=False)
        return path

    def filter_from_catalog(self, catalog: pd.DataFrame, merged_anomalies: Optional[dict] = None,
                            detections_df: Optional[pd.DataFrame] = None, start_col: str = "TIME",
                            stop_col: str = "END_TIME") -> tuple[pd.DataFrame, dict]:
        """
        Keep the events whose time interval overlaps at least one catalog entry.

        Naive timestamps, in the catalog or in the data, are taken as UTC.

        Parameters
        ----------
        - catalog (pd.DataFrame): Catalog with one row per known event.
        - merged_anomalies (Optional[dict]): Events as returned by
          :meth:`identify_and_merge_triggers`. Defaults to the last computed ones.
        - detections_df (Optional[pd.DataFrame]): Events summary from :meth:`get_detections_df`.
          Defaults to a fresh one.
        - start_col (str): Catalog column with the start of each entry.
        - stop_col (str): Catalog column with the end of each entry.

        Returns
        -------
        - tuple[pd.DataFrame, dict]: The matched rows of ``detections_df`` with a
          ``catalog_triggers`` column (list of catalog records), and the matched events with the
          same records added to every channel, keyed as in ``merged_anomalies``.

        Raises
        ------
        - ValueError: If the catalog is empty or misses ``start_col`` / ``stop_col``.
        """
        if catalog is None or catalog.empty:
            raise ValueError("catalog is None or empty.")
        for col in (start_col, stop_col):
            if col not in catalog.columns:
                raise ValueError(f"catalog has no column {col!r}.")
        if merged_anomalies is None:
            merged_anomalies = self.merged_anomalies
        if detections_df is None:
            detections_df = self.get_detections_df()

        cat_start = pd.to_datetime(catalog[start_col], utc=True).to_numpy()
        cat_stop = pd.to_datetime(catalog[stop_col], utc=True).to_numpy()
        starts = pd.to_datetime(detections_df[f"start_{self.time_col}"], utc=True).to_numpy()
        stops = pd.to_datetime(detections_df[f"stop_{self.time_col}"], utc=True).to_numpy()
        overlap = (starts[:, None] <= cat_stop[None, :]) & (stops[:, None] >= cat_start[None, :])

        records = [catalog.iloc[np.flatnonzero(row)].to_dict("records") for row in overlap]
        matched = detections_df.copy()
        matched["catalog_triggers"] = records
        matched = matched[overlap.any(axis=1)].reset_index(drop=True)

        results = {}
        for event_id, triggers in zip(matched["event_id"], matched["catalog_triggers"]):
            if event_id not in merged_anomalies:
                continue
            event = copy.deepcopy(merged_anomalies[event_id])
            for info in event.values():
                info["catalog_triggers"] = triggers
            results[event_id] = event
        return matched, results

    def plot_anomalies(self, merged_anomalies: Optional[dict] = None, return_df: Optional[pd.DataFrame] = None,
                       support_vars: Optional[Sequence[str]] = None, show: bool = False) -> None:
        """
        Plot every event with signals, background, significance and support variables.

        Parameters
        ----------
        - merged_anomalies (Optional[dict]): Events to plot. Defaults to the last computed ones.
        - return_df (Optional[pd.DataFrame]): Per-sample results. Defaults to ``results``.
        - support_vars (Optional[Sequence[str]]): Extra columns of ``tiles_df`` to plot.
        - show (bool): Show the figures interactively.
        """
        from .plotter import Plotter

        merged_anomalies = self.merged_anomalies if merged_anomalies is None else merged_anomalies
        return_df = self.results if return_df is None else return_df
        support_vars = list(support_vars or [])
        base = self.tiles_df[self.y_cols + self.y_cols_pred + support_vars + [self.time_col]].reset_index(drop=True)
        if self.std_cols:
            std = self.tiles_df[self.std_cols].reset_index(drop=True)
            std.columns = [f"{c}_std" for c in self.y_cols]
            base = pd.concat([base, std], axis=1)
        sig = return_df[[f"{c}_significance" for c in self.y_cols]].reset_index(drop=True)
        plot_df = pd.concat([base, sig], axis=1).rename(columns={self.time_col: "datetime"})
        Plotter(df=merged_anomalies).plot_anomalies(
            self.trigger_type, support_vars, self.thresholds, plot_df, self.y_cols, self.y_cols_pred,
            show=show, units=self.units, latex_y_cols=self.latex_y_cols,
        )
