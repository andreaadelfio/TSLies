"""
Helpers for ``trigger_validation.ipynb``: synthetic Gaussian data, the detectors under test,
event counting, and plots in the style of ``tslies.plotter``.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm

from tslies.benchmark import gaussian_dataset, profile
from tslies.stats import gaussian_focus

BACKGROUND = 100.0  # counts/s, the same in every channel
NOISE_STD = 10.0  # counts/s
MERGE_SECONDS = 60  # crossings closer than this are one event

# the detectors compared in the notebook: residuals of one channel -> (significance per second,
# length of the stretch behind it, or None when it is always one second)
DETECTORS = {
    "z-score": lambda z, resets, threshold: (np.maximum(z, 0.0), None),
    "FOCuS, no restarts": lambda z, resets, threshold: _with_length(gaussian_focus(z, resets=resets)),
    "FOCuS, restart after each alarm":
        lambda z, resets, threshold: _with_length(gaussian_focus(z, resets=resets, restart_above=threshold)),
    "FOCuS, alarmed points set to background":
        lambda z, resets, threshold: _with_length(gaussian_focus(z, resets=resets, background_above=threshold)),
}
# their significance depends on the threshold
NEEDS_THRESHOLD = ("FOCuS, restart after each alarm", "FOCuS, alarmed points set to background")
DETECTOR_COLORS = {"z-score": "tab:gray", "FOCuS, no restarts": "tab:blue",
                   "FOCuS, restart after each alarm": "tab:purple",
                   "FOCuS, alarmed points set to background": "tab:brown"}
KIND_COLORS = {"spike": "tab:red", "box": "tab:orange", "fred": "tab:green", "gaussian": "tab:cyan",
               "triangle": "tab:pink"}


# ------------------------------------------------------------------------------------------ data

def make_gaussian_dataset(n_channels, days, seed=0, start="2024-01-01"):
    """
    Synthetic data, one sample per second: every channel is a constant background plus
    independent Gaussian noise, ``y = BACKGROUND + NOISE_STD * z`` with ``z ~ N(0, 1)``
    (see :func:`tslies.benchmark.gaussian_dataset`).
    """
    return gaussian_dataset(n_channels, days, seed=seed, start=start, background=BACKGROUND, noise_std=NOISE_STD)


def residuals(df, cols):
    """``(y - background) / std`` of each channel, as an array (channels, samples)."""
    return np.vstack([((df[c] - df[f"{c}_bkg"]) / df[f"{c}_std"]).to_numpy() for c in cols])


def latex_labels(cols):
    """Labels for ``tslies.plotter``: ``ch3`` -> ``C_{3}``."""
    return {col: f"C_{{{col[2:]}}}" for col in cols}


def units(cols):
    """Units for ``tslies.plotter``."""
    return {col: "counts/s" for col in cols}


# ------------------------------------------------------------------------------------ detectors

def _with_length(result):
    """``(significance, length)`` of a :class:`tslies.stats.FocusResult`."""
    return result.significance, result.length


def significance(detector, z, resets=None, threshold=None, with_length=False):
    """
    Significance of one channel for one detector; ``threshold`` is needed by the detectors that
    change their memory after each alarm (``NEEDS_THRESHOLD``). With ``with_length``, also the
    length of the stretch behind each value (1 for the z-score).

    Non-finite residuals are gaps: their significance is NaN and the detector restarts after them,
    as in ``tslies.trigger.Trigger``.
    """
    if detector in NEEDS_THRESHOLD and threshold is None:
        raise ValueError(f"{detector!r} needs a threshold.")
    z = np.asarray(z, dtype=float)
    resets = np.zeros(len(z), dtype=bool) if resets is None else np.asarray(resets, dtype=bool)
    out = np.full(len(z), np.nan)
    length = np.zeros(len(z), dtype=np.int64) if with_length else None
    idx = np.flatnonzero(np.isfinite(z))
    if idx.size:
        sub_resets = resets[idx].copy()
        sub_resets[1:] |= (np.diff(idx) > 1) | (np.diff(np.cumsum(resets)[idx]) > 0)
        out[idx], sub_length = DETECTORS[detector](z[idx], sub_resets, threshold)
        if with_length:
            length[idx] = 1 if sub_length is None else sub_length
    return (out, length) if with_length else out


def detector_events(detector, z, threshold, resets=None, merge=MERGE_SECONDS):
    """
    Events found by one detector on one channel, as ``tslies.trigger.Trigger`` finds them: each
    alarm marks the stretch that raised it, from the estimated start of the change to the alarm,
    and marked stretches closer than ``merge`` seconds are one event.

    Returns
    -------
    - list[tuple[int, int]]: ``(start, stop)`` sample positions of each event.
    """
    sig, length = significance(detector, z, resets, threshold, with_length=True)
    alarms = np.flatnonzero(np.nan_to_num(sig, nan=0.0) > threshold)
    edges = np.zeros(len(sig) + 1, dtype=np.int64)
    np.add.at(edges, alarms - length[alarms] + 1, 1)
    np.add.at(edges, alarms + 1, -1)
    return event_spans(np.cumsum(edges[:-1]) > 0, 0.5, resets, merge)


def grouped_events(detector, Z, threshold, resets=None, merge=MERGE_SECONDS):
    """
    Events of one detector on all the channels (``Z``: channels, samples), grouped as
    ``tslies.trigger.Trigger`` groups them: each alarm, in any channel, marks the stretch that
    raised it (see :func:`detector_events`), and marked stretches closer than ``merge`` seconds
    are one event.
    """
    marked = np.zeros(Z.shape[1], dtype=bool)
    for z in Z:
        for start, stop in detector_events(detector, z, threshold, resets, merge=0):
            marked[start:stop + 1] = True
    return event_spans(marked, 0.5, resets, merge)


def run_detector(detector, Z, cols, channel_sets, resets=None, threshold=None):
    """
    Run a detector on every channel and keep, for each named set of channels, the maximum
    significance over its channels at every second: the trigger fires when any of them is over
    threshold.

    Parameters
    ----------
    - detector (str): Key of ``DETECTORS``.
    - Z (np.ndarray): Residuals, shape (channels, samples), in the order of ``cols``.
    - cols (list[str]): Channel names.
    - channel_sets (dict[str, list[str]]): Named sets of channels.
    - resets (Optional[np.ndarray]): Detector restarts.
    - threshold (Optional[float]): For the detectors that restart after each alarm.

    Returns
    -------
    - dict[str, np.ndarray]: Maximum significance per set of channels at every second.
    """
    maxima = {name: np.zeros(Z.shape[1], dtype=np.float32) for name in channel_sets}
    for i, col in enumerate(cols):
        filled = np.nan_to_num(significance(detector, Z[i], resets, threshold), nan=0.0).astype(np.float32)
        for name, members in channel_sets.items():
            if col in members:
                np.maximum(maxima[name], filled, out=maxima[name])
    return maxima


def measure_isolated(detector, Z, cols, truth, before=600, after=MERGE_SECONDS, threshold=None):
    """
    Highest significance reached by each anomaly in its strongest channel, with the detector
    started ``before`` seconds before the anomaly, so that earlier anomalies do not interfere.
    """
    out = np.full(len(truth), np.nan)
    for j, (col, a, b) in enumerate(zip(truth["strongest"], truth["start_index"], truth["stop_index"])):
        first = max(a - before, 0)
        sig = significance(detector, Z[cols.index(col), first:b + after + 1], threshold=threshold)
        out[j] = np.nanmax(sig[a - first:])
    return out


def pick_examples(events, truth, kinds, strength=12.0):
    """
    Among the events matched to injected signals by ``Trigger.filter_from_catalog`` (catalog
    names ``<kind>_<anomaly_id>``), one per kind, the one whose signal strength is closest to
    ``strength``.
    """
    picked = {}
    for kind in kinds:
        candidates = {}
        for event_id, info in events.items():
            name = next(iter(info.values()))["catalog_triggers"][0]["NAME"]
            if name.rsplit("_", 1)[0] == kind:
                candidates[event_id] = truth["z_peak_channel"].iat[int(name.rsplit("_", 1)[1])]
        if candidates:
            best = min(candidates, key=lambda e: abs(np.log(candidates[e] / strength)))
            picked[best] = events[best]
    return picked


def brute_force_focus(z):
    """
    FOCuS significance computed from its definition, for comparison: at every second all the
    stretches ending there are tried one by one, so the cost grows like the square of the length.
    """
    csum = np.concatenate(([0.0], np.cumsum(z)))
    out = np.zeros(len(z))
    for t in range(1, len(z) + 1):
        total = csum[t] - csum[:t]
        out[t - 1] = np.sqrt(np.max(np.where(total > 0, total * total / (t - np.arange(t)), 0.0)))
    return out


def event_spans(max_sig, threshold, resets=None, merge=MERGE_SECONDS):
    """
    Events as ``(first, last)`` sample positions: runs of seconds over threshold, merged when
    closer than ``merge`` seconds (and never across a reset).
    """
    above = np.asarray(max_sig) > threshold
    edges = np.diff(np.r_[0, above.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if starts.size == 0:
        return []
    segment = np.zeros(len(above), dtype=int) if resets is None else np.cumsum(resets)
    new_event = np.r_[True, (starts[1:] - stops[:-1] - 1 > merge) | (segment[starts[1:]] != segment[stops[:-1]])]
    firsts = starts[new_event]
    lasts = stops[np.r_[np.flatnonzero(new_event)[1:] - 1, len(starts) - 1]]
    return list(zip(firsts.tolist(), lasts.tolist()))


def false_alarms_per_day(max_sig, thresholds, resets=None, merge=MERGE_SECONDS):
    """Events per day at every threshold, on data without anomalies (so every event is false)."""
    days = len(max_sig) / 86_400
    return np.array([len(event_spans(max_sig, t, resets, merge)) for t in thresholds]) / days


def false_alarms_by_threshold(detector, Z, cols, channel_sets, thresholds, resets=None):
    """
    False alarms per day of a detector whose significance depends on the threshold (one of
    ``NEEDS_THRESHOLD``), on data without anomalies: it is run again for every threshold.

    Returns
    -------
    - dict[str, np.ndarray]: For each set of channels, the false alarms per day at each threshold.
    """
    rates = {name: [] for name in channel_sets}
    for t in thresholds:
        maxima = run_detector(detector, Z, cols, channel_sets, resets=resets, threshold=t)
        for name, max_sig in maxima.items():
            rates[name].append(false_alarms_per_day(max_sig, [t], resets)[0])
    return {name: np.array(values) for name, values in rates.items()}


def threshold_at_rate(thresholds, rates, rate=1.0):
    """
    Lowest threshold above which there are at most ``rate`` false alarms per day (NaN if none).

    The rate is not monotonic: at very low thresholds the trigger is on almost all the time and
    the crossings merge into a few long events. So the threshold is taken after the last one
    with too many false alarms, not at the first one with few.
    """
    thresholds = np.asarray(thresholds)
    too_many = np.flatnonzero(np.asarray(rates) > rate)
    if too_many.size == 0:
        return float(thresholds[0])
    return float(thresholds[too_many[-1] + 1]) if too_many[-1] + 1 < len(thresholds) else np.nan


def detected(max_sig, truth, threshold, resets=None, after=MERGE_SECONDS, merge=MERGE_SECONDS):
    """
    For each anomaly: does a new event start between its start and ``after`` seconds after its
    end? A trigger that is already on because of an earlier event does not count.
    """
    starts = np.array([a for a, _ in event_spans(max_sig, threshold, resets, merge)], dtype=int)
    return np.array([bool(((starts >= a) & (starts <= b + after)).any())
                     for a, b in zip(truth["start_index"], truth["stop_index"])])


def false_alarms_with_signals(max_sig, truth, threshold, resets=None, after=MERGE_SECONDS, merge=MERGE_SECONDS):
    """Events per day that do not start during an anomaly (or within ``after`` seconds from its end)."""
    starts = np.array([a for a, _ in event_spans(max_sig, threshold, resets, merge)], dtype=int)
    inside = np.zeros(len(starts), dtype=bool)
    for a, b in zip(truth["start_index"], truth["stop_index"]):
        inside |= (starts >= a) & (starts <= b + after)
    return (~inside).sum() / (len(max_sig) / 86_400)


# ---------------------------------------------------------------------------------------- plots

def style_axis(ax, scientific_y=False):
    """Dashed grid and, optionally, a scientific y axis, as in ``Plotter.plot_anomalies``."""
    ax.grid(ls="--", alpha=0.6)
    if scientific_y:
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)


def time_axis(ax, t):
    """Readable time ticks for a time range of any length, and the start time in the label."""
    t = pd.DatetimeIndex(t)
    span = (t[-1] - t[0]).total_seconds()
    step = next((s for s in (10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200) if span / s <= 6), 21_600)
    if step < 60:
        ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=range(0, 60, step)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    elif step < 3600:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, step // 60)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, step // 3600)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))
    ax.set_xlabel(f"datetime ({t[0]:%Y-%m-%d %H:%M:%S})")


def plot_data_example(df, col, start, stop):
    """Rate, background and noise band of one channel, and its residuals below, as in ``Plotter.plot_tile``."""
    part = df.iloc[start:stop]
    t = part["datetime"]
    fig, (ax_signal, ax_res) = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True, gridspec_kw={"hspace": 0})
    ax_signal.fill_between(t, part[f"{col}_bkg"] - part[f"{col}_std"], part[f"{col}_bkg"] + part[f"{col}_std"],
                           alpha=0.3, label=r"background $\pm$ noise std")
    ax_signal.plot(t, part[col], color="black", lw=0.6, label=col)
    ax_signal.plot(t, part[f"{col}_bkg"], color="red", label="background")
    ax_signal.set_ylabel("[counts/s]")
    z = (part[col] - part[f"{col}_bkg"]) / part[f"{col}_std"]
    ax_res.plot(t, z, color="black", lw=0.6, label="residuals z")
    ax_res.axhline(0, color="red")
    ax_res.set_ylabel(r"$z$ [$\sigma$]")
    for ax in (ax_signal, ax_res):
        style_axis(ax)
        ax.legend(loc="upper right")
    time_axis(ax_res, t)
    fig.tight_layout()
    return fig


def plot_significance_panels(t, z, significances, thresholds, title, events=None, signal=None):
    """
    Residuals on top and, below, one significance panel per detector with its threshold.

    Optionally, as in ``Plotter.plot_anomalies``: the ``(start, stop)`` of the true ``signal``
    (green lines, yellow band) in every panel, and the ``events`` found by each detector
    (``{name: [(start, stop), ...]}``, red lines and band) in its own panel.
    """
    n = 1 + len(significances)
    fig, axs = plt.subplots(n, 1, figsize=(8.5, 1.7 * n + 1.3), sharex=True, gridspec_kw={"hspace": 0})
    axs[0].plot(t, z, color="black", lw=0.6, label="residuals z")
    axs[0].axhline(0, color="red")
    axs[0].set_ylabel(r"$z$ [$\sigma$]")
    axs[0].set_title(title)
    for k, ax in enumerate(axs if signal is not None else []):
        ax.axvspan(t[signal[0]], t[signal[1]], color="yellow", alpha=0.1)
        ax.axvline(t[signal[0]], color="green", lw=0.8, label="true signal" if k == 0 else None)
        ax.axvline(t[signal[1]], color="green", lw=0.8)
    for ax, (name, sig) in zip(axs[1:], significances.items()):
        ax.plot(t, sig, color="blue", lw=0.7, label=f"S: {name}")
        ax.axhline(thresholds[name], color="darkorange", ls="-.", label=f"threshold {thresholds[name]:g}")
        for j, (start, stop) in enumerate((events or {}).get(name, [])):
            ax.axvspan(t[start], t[stop], color="red", alpha=0.1)
            ax.axvline(t[start], color="red", lw=0.8, label="events found" if j == 0 else None)
            ax.axvline(t[stop], color="red", lw=0.8)
        ax.set_ylabel("significance")
        ax.set_ylim(0, max(thresholds[name] * 1.25, np.nanmax(sig) * 1.1))
    for ax in axs:
        style_axis(ax)
        ax.legend(loc="upper right", fontsize=8)
    time_axis(axs[-1], t)
    fig.tight_layout()
    return fig


def line_style(name):
    """FOCuS with restarts is drawn dashed and thinner, so that FOCuS without restarts stays visible below it."""
    return {"lw": 1.3, "ls": "--"} if name in NEEDS_THRESHOLD else {"lw": 2}


def plot_exceedance(levels, significances):
    """Fraction of the seconds with significance above each level, against the Gaussian expectation."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, sig in significances.items():
        fraction = np.array([np.mean(sig > level) for level in levels])
        ax.plot(levels, np.where(fraction > 0, fraction, np.nan), color=DETECTOR_COLORS[name], label=name,
                **line_style(name))
    ax.plot(levels, norm.sf(levels), "k--", lw=1, label=r"$P(z > \mathrm{level})$, standard Gaussian")
    ax.set_yscale("log")
    ax.set_ylim(np.nanmin([1 / len(s) for s in significances.values()]) / 2, 1)
    ax.set_xlabel("level")
    ax.set_ylabel("fraction of the seconds with significance > level")
    style_axis(ax)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_false_alarms(rates, chosen, theory=None, title=""):
    """
    False alarms per day against the threshold, one line per detector: ``rates`` and ``theory``
    are ``pd.Series`` indexed by threshold (FOCuS with restarts is computed at fewer thresholds).
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, values in rates.items():
        ax.plot(values.index, values.where(values > 0), color=DETECTOR_COLORS[name],
                label=f"{name}: 1 per day at {chosen[name]:.2f}", **line_style(name))
        ax.plot(chosen[name], 1.0, "o", color=DETECTOR_COLORS[name], ms=7, mec="white")
    if theory is not None:
        ax.plot(theory.index, theory.where(theory > 1e-3), "k--", lw=1, label="z-score, Gaussian expectation")
    ax.axhline(1, color="darkorange", ls="-.", label="1 false alarm per day")
    first = min(values.index.min() for values in rates.values())
    last = max(values[values > 0].index.max() for values in rates.values() if (values > 0).any())
    ax.set_xlim(first, last + 1)
    ax.set_ylim(1e-2, None)
    ax.set_yscale("log")
    ax.set_xlabel("threshold on the significance")
    ax.set_ylabel("false alarms per day")
    ax.set_title(title)
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_anomaly_shapes(examples):
    """The shapes of the injected anomalies, e.g. ``{'fred': 120, ...}`` (kind: duration in s)."""
    fig, axs = plt.subplots(1, len(examples), figsize=(3 * len(examples), 2.6), sharey=True)
    for ax, (kind, duration) in zip(axs, examples.items()):
        pad = max(duration // 3, 5)
        t = np.arange(-pad, duration + pad)
        y = np.zeros(len(t))
        y[pad:pad + duration] = profile(kind, duration, rise_fraction=0.1)
        ax.step(t, y, where="mid", color="green")
        ax.set_title(f"{kind}, {duration} s")
        ax.set_xlabel("seconds from the start")
        style_axis(ax)
    axs[0].set_ylabel("shape (peak = 1)")
    fig.tight_layout()
    return fig


def plot_injection_examples(times, Z, Z_inj, cols, truth, thresholds, resets=None, strength=10.0):
    """
    For each kind, the anomaly whose strength is closest to ``strength``: the residuals of its
    strongest channel with the injected shape, and below the significance of every detector.
    """
    resets = np.zeros(Z.shape[1], dtype=bool) if resets is None else resets
    kinds = [k for k in KIND_COLORS if k in set(truth["kind"])]
    fig, axs = plt.subplots(2, len(kinds), figsize=(4.3 * len(kinds), 6.2), sharex="col", gridspec_kw={"hspace": 0})
    segment_starts = np.unique(np.r_[0, np.flatnonzero(resets)])
    for k, kind in enumerate(kinds):
        candidates = truth[truth["kind"] == kind]
        row = candidates.iloc[int(np.argmin(np.abs(np.log(candidates["z_peak_channel"] / strength))))]
        ch = cols.index(row["strongest"])
        a, b = int(row["start_index"]), int(row["stop_index"])
        pad = max(120, b - a)
        lo, hi = max(a - pad, 0), min(b + pad, Z.shape[1] - 1)
        first = segment_starts[segment_starts <= a].max()  # the detectors need the whole history
        t = times[lo:hi + 1]
        ax = axs[0, k]
        ax.plot(t, Z_inj[ch, lo:hi + 1], color="black", lw=0.6, label="residuals z with the anomaly")
        ax.plot(t, Z_inj[ch, lo:hi + 1] - Z[ch, lo:hi + 1], color="green", lw=1.5, label="injected anomaly")
        ax.set_title(f"{kind}: {row['strongest']}, strength {row['z_peak_channel']:.1f}, {row['duration']} s", fontsize=10)
        ax2 = axs[1, k]
        for name in DETECTORS:
            sig = significance(name, Z_inj[ch, first:hi + 1], resets[first:hi + 1], thresholds[name])[lo - first:]
            ax2.plot(t, sig, color=DETECTOR_COLORS[name], lw=0.9, label=name)
            ax2.axhline(thresholds[name], color=DETECTOR_COLORS[name], ls="-.", lw=0.8)
        for axis in (ax, ax2):
            axis.axvspan(times[a], times[b], color="yellow", alpha=0.1)
            axis.axvline(times[a], color="green", lw=0.8)
            axis.axvline(times[b], color="green", lw=0.8)
            style_axis(axis)
        time_axis(ax2, t)
    axs[0, 0].set_ylabel(r"$z$ [$\sigma$]")
    axs[1, 0].set_ylabel("significance\n(dash-dot: threshold)")
    axs[0, 0].legend(loc="upper left", fontsize=7)
    axs[1, 0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    return fig


def plot_expected_vs_measured(truth, measured, title, thresholds=None):
    """
    Highest significance reached during each anomaly (strongest channel) against its strength,
    with the threshold of each detector (dash-dot) when ``thresholds`` is given.
    """
    names = list(measured)
    fig, axs = plt.subplots(1, len(names), figsize=(4 * len(names), 4.1), sharex=True, sharey=True)
    strength = truth["z_peak_channel"].to_numpy()
    for ax, name in zip(axs, names):
        for kind, color in KIND_COLORS.items():
            sel = (truth["kind"] == kind).to_numpy()
            ax.scatter(strength[sel], np.maximum(measured[name][sel], 0.9), s=16, color=color, label=kind,
                       edgecolor="white", linewidth=0.4)
        ax.plot([1, 100], [1, 100], "k--", lw=1, label="significance = strength")
        if thresholds is not None:
            ax.axhline(thresholds[name], color="darkorange", ls="-.", lw=1, label="threshold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(2.5, 40)
        ax.set_ylim(0.9, 150)
        ax.set_title(name)
        ax.set_xlabel("strength of the injected anomaly")
        style_axis(ax)
    axs[0].set_ylabel("highest significance during the anomaly")
    axs[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_found_fraction(truth, found, bins=(3, 5, 8, 12, 18, 30)):
    """Fraction of the anomalies found against their strength, one line per detector."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    edges = np.asarray(bins, dtype=float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    strength = truth["z_peak_channel"].to_numpy()
    for name, hit in found.items():
        fraction = [hit[(strength >= lo) & (strength < hi)].mean() * 100 for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, fraction, "-o", color=DETECTOR_COLORS[name], lw=2, ms=6,
                label=f"{name}: {hit.mean() * 100:.0f}% overall")
    ax.set_xscale("log")
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:g}" for e in edges])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(-3, 103)
    ax.set_xlabel("strength of the injected anomaly")
    ax.set_ylabel("anomalies found [%]")
    style_axis(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)  # above the lines
    fig.tight_layout()
    return fig


def plot_timing(truth_out, title):
    """How early each anomaly is detected and how well its start is estimated, by kind."""
    kinds = [k for k in KIND_COLORS if k in set(truth_out["kind"])]
    found = truth_out[truth_out["detected"]]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    rng = np.random.default_rng(0)
    for k, kind in enumerate(kinds):
        sel = found[found["kind"] == kind]
        x = k + rng.uniform(-0.2, 0.2, len(sel))
        axs[0].scatter(x, np.maximum(sel["delay"], 0.5), s=16, color=KIND_COLORS[kind])
        axs[0].scatter(x, sel["duration"], s=10, marker="_", color="black")
        axs[1].scatter(x, sel["start_error"], s=16, color=KIND_COLORS[kind])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("seconds from the start of the anomaly\nto the first second over threshold")
    axs[0].set_title("detection delay (black dash: duration of the anomaly)")
    axs[1].axhline(0, color="red")
    axs[1].set_yscale("symlog", linthresh=10)
    axs[1].set_ylabel("estimated start - true start [s]")
    axs[1].set_title("error on the start of the anomaly")
    for ax in axs:
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds)
        style_axis(ax)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_compute_time(seconds):
    """Computing time against the length of the data, one line per method (``{method: {length: s}}``)."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, values in seconds.items():
        lengths = sorted(values)
        ax.plot(lengths, [values[n] for n in lengths], "-o", lw=2, ms=5,
                color=DETECTOR_COLORS.get(name, "black"), label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("length of the data [samples], one channel")
    ax.set_ylabel("computing time [s]")
    style_axis(ax)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig
