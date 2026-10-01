"""
Long benchmark and non-regression test of the trigger.

Twelve weeks of synthetic data (5 channels, one sample per second, Gaussian noise around a known
background), each with 84 injected anomalies of the five kinds of :mod:`tslies.benchmark` (1008 in
all), run through :class:`tslies.trigger.Trigger` as in an analysis, with Gaussian FOCuS at the
threshold for 1 false alarm per day found in ``tslies/examples/example4/trigger_validation.ipynb``.
Two methods are compared: FOCuS restarting after each alarm, and FOCuS keeping its memory with the
alarmed samples set to background (``after_alarm`` of ``Trigger``). Every week is also run
without the anomalies, to count the false alarms on pure noise.

Everything is generated from fixed seeds, so the results change only if the code changes (or if
numpy's random numbers change, which the checksum of the data detects). They are compared with
``long_benchmark_reference.json``, the results last judged good.

Usage, from the repository root::

    python benchmarks/long_benchmark.py           # run (about 7 minutes) and compare with the reference
    python benchmarks/long_benchmark.py --accept  # the results of the last run become the reference

The results of the last run are always written to ``long_benchmark_last.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tslies  # noqa: F401
except ModuleNotFoundError:  # tslies is not installed: use this repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tslies.benchmark import gaussian_dataset, inject, match_events, random_anomalies
from tslies.trigger import Trigger

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "long_benchmark_reference.json"
LAST = HERE / "long_benchmark_last.json"

SETTINGS = {
    "weeks": 12,
    "days_per_week": 7,
    "channels": 5,
    "anomalies_per_week": 84,  # 12 per day
    "seed": 2024,
    "strength_range": [3, 30],
    "channels_per_anomaly": [1, 3],
    "min_gap": 600,
    "margin": 120,
    "merge_interval": 60,
    "threshold": 5.25,  # 1 false alarm per day on 5 channels
    "methods": {  # what FOCuS does after an alarm (after_alarm of Trigger)
        "FOCuS, restart after each alarm": "restart",
        "FOCuS, alarmed points set to background": "background",
    },
    "strength_bins": [3, 5, 8, 12, 18, 30],
}


def week(i: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Week ``i``: the pure noise, the same noise with the anomalies, the anomalies, the channels."""
    s = SETTINGS
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=s["days_per_week"] * i)
    noise, cols = gaussian_dataset(s["channels"], s["days_per_week"], seed=s["seed"] + i, start=str(start))
    anomalies = random_anomalies(s["anomalies_per_week"], len(noise), cols, n_groups=tuple(s["channels_per_anomaly"]),
                                 z_range=tuple(s["strength_range"]), margin=s["margin"], min_gap=s["min_gap"],
                                 seed=s["seed"] + 1000 + i)
    data, truth = inject(noise, anomalies, std_suffix="_std")
    return noise, data, truth, cols


def detections(df: pd.DataFrame, cols: list[str], after_alarm: str) -> pd.DataFrame:
    """Events found by ``Trigger`` on ``df``, with the true background and noise as background model."""
    trigger = Trigger(df, cols, [f"{c}_bkg" for c in cols], thresholds=SETTINGS["threshold"], trigger_type="focus",
                      std_cols=[f"{c}_std" for c in cols], after_alarm=after_alarm)
    trigger.run()
    trigger.identify_and_merge_triggers(merge_interval=SETTINGS["merge_interval"])
    return trigger.get_detections_df()


def summarize(signals: pd.DataFrame, counts: dict, duration_ratios: list, longest: int, days: float) -> dict:
    """The numbers of one method, rounded so that they can be compared between runs."""
    bins = SETTINGS["strength_bins"]
    strength = pd.cut(signals["z_peak_channel"], bins, right=False,
                      labels=[f"{lo}-{hi}" for lo, hi in zip(bins[:-1], bins[1:])])
    found = signals[signals["detected"]]

    def r(x):
        return round(float(x), 4)

    return {
        "signals": len(signals),
        "found": int(signals["detected"].sum()),
        "found [%]": r(signals["detected"].mean() * 100),
        "found [%] by kind": {k: r(v * 100) for k, v in signals.groupby("kind")["detected"].mean().items()},
        "found [%] by strength": {str(k): r(v * 100) for k, v in
                                  signals.groupby(strength, observed=False)["detected"].mean().items()},
        "events": counts["events"],
        "false alarms": counts["false alarms"],
        "false alarms per day": r(counts["false alarms"] / days),
        "false alarms per day, pure noise": r(counts["noise events"] / days),
        "events with more than one signal": counts["shared events"],
        "signals in events with more than one signal": counts["shared signals"],
        "longest event [hours]": r(longest / 3600),
        "found before their end [%]": r((found["delay"] < found["duration"]).mean() * 100),
        "median delay [s] by kind": {k: r(v) for k, v in found.groupby("kind")["delay"].median().items()},
        "median start error [s] by kind": {k: r(v) for k, v in found.groupby("kind")["start_error"].median().items()},
        "median event duration / signal duration": r(np.median(duration_ratios)),
    }


def run(verbose: bool = True) -> dict:
    """Run the benchmark; the results can be saved as JSON."""
    s = SETTINGS
    methods = s["methods"]
    checksum = hashlib.sha256()
    signals = {m: [] for m in methods}
    counts = {m: dict.fromkeys(["noise events", "events", "false alarms", "shared events", "shared signals"], 0)
              for m in methods}
    duration_ratios = {m: [] for m in methods}
    longest = dict.fromkeys(methods, 0)
    seconds = dict.fromkeys(methods, 0.0)
    for i in range(s["weeks"]):
        noise, data, truth, cols = week(i)
        checksum.update(np.ascontiguousarray(data[cols].to_numpy()).tobytes())
        durations = truth.set_index("anomaly_id")["duration"]
        for m, after_alarm in methods.items():
            t0 = time.perf_counter()
            noise_events = detections(noise, cols, after_alarm)
            events = detections(data, cols, after_alarm)
            seconds[m] += time.perf_counter() - t0
            truth_out, events_out, summary = match_events(events, truth)
            n_signals = events_out["anomaly_ids"].str.len()
            counts[m]["noise events"] += len(noise_events)
            counts[m]["events"] += summary["n_events"]
            counts[m]["false alarms"] += summary["n_false"]
            counts[m]["shared events"] += int((n_signals > 1).sum())
            counts[m]["shared signals"] += int(n_signals[n_signals > 1].sum())
            if len(events):
                longest[m] = max(longest[m], int((events["stop_index"] - events["start_index"] + 1).max()))
            signals[m].append(truth_out[["kind", "z_peak_channel", "duration", "detected", "delay", "start_error"]])
            single = events_out[n_signals == 1]  # events with exactly one signal
            lengths = (single["stop_index"] - single["start_index"] + 1).to_numpy()
            duration_ratios[m].extend(lengths / durations.loc[[ids[0] for ids in single["anomaly_ids"]]].to_numpy())
        if verbose:
            print(f"week {i + 1} of {s['weeks']} done", flush=True)
    days = s["weeks"] * s["days_per_week"]
    return {
        "settings": s,
        "data_sha256": checksum.hexdigest(),
        "results": {m: summarize(pd.concat(signals[m], ignore_index=True), counts[m], duration_ratios[m],
                                 longest[m], days) for m in methods},
        "seconds": {m: round(v, 1) for m, v in seconds.items()},  # not compared: they depend on the computer
    }


def flatten(d: dict, prefix: str = "") -> dict:
    """``{"a": {"b": 1}}`` -> ``{"a / b": 1}``."""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flatten(value, f"{prefix}{key} / "))
        else:
            out[f"{prefix}{key}"] = value
    return out


def compare(current: dict, reference: dict) -> tuple[list[str], bool]:
    """The lines of a report on what differs from the reference, and whether nothing does."""
    lines = []
    if current["settings"] != reference["settings"]:
        lines.append("WARNING: the settings differ from those of the reference, the results cannot be compared.")
    if current["data_sha256"] != reference["data_sha256"]:
        lines.append("WARNING: the data differ from those of the reference (another numpy version?), "
                     "the results cannot be compared.")
    cur, ref = flatten(current["results"]), flatten(reference["results"])
    for key in dict.fromkeys([*ref, *cur]):  # the keys of both, in order
        if cur.get(key) != ref.get(key):
            lines.append(f"  {key}: {ref.get(key)} -> {cur.get(key)}")
    return lines, not lines


def report(results: dict) -> str:
    """The main numbers of a run, one column per method."""
    table = pd.DataFrame({m: flatten(r) for m, r in results["results"].items()})
    table.loc["computing time [s]"] = [results["seconds"][m] for m in table.columns]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        return table.to_string()


def load_reference() -> dict | None:
    """The reference results, or None if there are none yet."""
    return json.loads(REFERENCE.read_text()) if REFERENCE.exists() else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Long benchmark and non-regression test of the trigger.")
    parser.add_argument("--accept", action="store_true", help="make the results of the last run the new reference")
    args = parser.parse_args(argv)
    if args.accept:
        if not LAST.exists():
            print(f"No results to accept: run the benchmark first ({LAST.name} is missing).")
            return 1
        shutil.copyfile(LAST, REFERENCE)
        print(f"{LAST.name} is now the reference ({REFERENCE.name}).")
        return 0

    current = run()
    LAST.write_text(json.dumps(current, indent=2) + "\n")
    print(report(current))
    reference = load_reference()
    if reference is None:
        print("\nNo reference yet: if these results are good, make them the reference with --accept.")
        return 0
    lines, same = compare(current, reference)
    print("\nSame results as the reference." if same else "\nDIFFERENT from the reference:\n" + "\n".join(lines))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
