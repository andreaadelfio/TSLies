import numpy as np
import pandas as pd
import pytest

from tslies.benchmark import (Anomaly, gaussian_dataset, inject, match_events, profile, random_anomalies,
                              rolling_normalize, segment_bounds)
from tslies.trigger import Trigger


def test_segment_bounds():
    assert segment_bounds(10) == [(0, 10)]
    resets = np.zeros(10, dtype=bool)
    resets[[0, 4, 7]] = True
    assert segment_bounds(10, resets) == [(0, 4), (4, 7), (7, 10)]
    with pytest.raises(ValueError, match="shape"):
        segment_bounds(10, [True])


def test_rolling_normalize_gives_unit_gaussian_residuals_per_segment():
    rng = np.random.default_rng(3)
    n = 20_000
    level = np.where(np.arange(n) < 10_000, 100.0, 300.0)  # a jump exactly at the reset
    noise = np.where(np.arange(n) < 10_000, 2.0, 5.0)
    df = pd.DataFrame({"a": level + noise * rng.standard_normal(n)})
    resets = np.zeros(n, dtype=bool)
    resets[10_000] = True
    mean, std = rolling_normalize(df, ["a"], window=600, resets=resets)
    z = (df["a"] - mean["a"]) / std["a"]
    assert z.notna().all()
    assert z.mean() == pytest.approx(0.0, abs=0.02) and z.std() == pytest.approx(1.0, abs=0.02)
    assert mean["a"].iloc[9_999] == pytest.approx(100.0, abs=0.5)  # nothing leaks across the reset
    assert mean["a"].iloc[10_000] == pytest.approx(300.0, abs=1.0)


def test_rolling_normalize_short_segments_and_validation():
    df = pd.DataFrame({"a": np.arange(100.0)})
    resets = np.zeros(100, dtype=bool)
    resets[90] = True  # a segment of 10 samples, shorter than window // 2
    mean, std = rolling_normalize(df, ["a"], window=50, resets=resets)
    assert mean["a"].iloc[90:].isna().all() and mean["a"].iloc[:90].notna().all()
    flat = pd.DataFrame({"a": np.ones(100)})
    assert rolling_normalize(flat, ["a"], window=10)[1]["a"].isna().all()
    with pytest.raises(ValueError, match="window"):
        rolling_normalize(df, ["a"], window=1)
    with pytest.raises(ValueError, match="min_periods"):
        rolling_normalize(df, ["a"], window=10, min_periods=20)


@pytest.mark.parametrize("kind,duration", [("spike", 1), ("box", 7), ("fred", 50), ("fred", 400),
                                           ("gaussian", 100), ("triangle", 60)])
def test_profiles_peak_at_one(kind, duration):
    p = profile(kind, duration, rise_fraction=0.1)
    assert p.shape == (duration,)
    assert p.max() == pytest.approx(1.0)
    assert np.all(p > 0)


def test_fred_rises_fast_and_decays_to_one_percent():
    p = profile("fred", 500, rise_fraction=0.1)
    assert abs(np.argmax(p) - 50) <= 2
    assert p[-1] == pytest.approx(0.01, rel=0.1)
    assert p[0] < 0.01


@pytest.mark.parametrize("args,match", [(("wave", 5), "kind"), (("box", 0), "positive"),
                                        (("spike", 3), "one sample"), (("fred", 10, 1.5), "rise_fraction")])
def test_profile_validation(args, match):
    with pytest.raises(ValueError, match=match):
        profile(*args)


def test_inject_adds_scaled_shapes_and_describes_them():
    n = 1000
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=n, freq="s"),
                       "a": np.zeros(n), "b": np.zeros(n), "a_std": 2.0, "b_std": 0.5})
    anomalies = [Anomaly("box", 100, 10, 3.0, ("a",)),
                 Anomaly("triangle", 500, 5, -4.0, ("a", "b"), (1.0, 0.5))]
    out, truth = inject(df, anomalies)
    np.testing.assert_allclose(out["a"].iloc[100:110], 6.0)
    np.testing.assert_allclose(out["b"].iloc[500:505], -4.0 * 0.5 * 0.5 * profile("triangle", 5))
    assert out["a"].iloc[:100].eq(0).all() and out["injected_id"].iloc[100:110].eq(0).all()
    assert out["injected_id"].iloc[500:505].eq(1).all() and out["injected_id"].iloc[110:500].eq(-1).all()
    np.testing.assert_allclose(out["a_injected"] + df["a"], out["a"])
    assert truth["z_peak_channel"].iat[0] == pytest.approx(3.0 * np.sqrt(10))
    assert truth["snr_matched"].iat[1] == pytest.approx(4.0 * np.sqrt(1.25 * np.sum(profile("triangle", 5) ** 2)))
    assert truth["peak_index"].tolist() == [100, 502]
    assert truth["start_datetime"].iat[0] == df["datetime"].iat[100]
    df_before = df.copy()
    with pytest.raises(ValueError, match="outside"):
        inject(df, [Anomaly("box", 995, 10, 1.0, ("a",))])
    with pytest.raises(ValueError, match="no column"):
        inject(df, [Anomaly("box", 10, 10, 1.0, ("c",))])
    pd.testing.assert_frame_equal(df, df_before)


def test_random_anomalies_respect_constraints():
    n = 50_000
    resets = np.zeros(n, dtype=bool)
    resets[[20_000, 35_000]] = True
    groups = {"f1": ["a1", "a2"], "f2": ["b1", "b2"], "f3": ["c1"]}
    channels = [c for g in groups.values() for c in g]
    anomalies = random_anomalies(40, n, channels, resets=resets, groups=groups, n_groups=(1, 2),
                                 negative_fraction=0.25, margin=50, min_gap=200, seed=4)
    assert len(anomalies) == 40 and [a.start for a in anomalies] == sorted(a.start for a in anomalies)
    bounds = segment_bounds(n, resets)
    for a in anomalies:
        assert any(lo + 50 <= a.start and a.stop <= hi - 51 for lo, hi in bounds)
        affected_groups = {g for g, members in groups.items() if set(members) & set(a.channels)}
        assert all(set(groups[g]) <= set(a.channels) for g in affected_groups)
        assert 1 <= len(affected_groups) <= 2 and max(a.weights) == pytest.approx(1.0)
    for first, second in zip(anomalies, anomalies[1:]):
        assert second.start - first.stop > 200
    df = pd.DataFrame({c: np.zeros(n) for c in channels} | {f"{c}_std": np.ones(n) for c in channels})
    _, truth = inject(df, anomalies, time_col=None)
    assert truth["z_peak_channel"].between(3.0 - 1e-9, 30.0 + 1e-9).all()
    assert (truth["amplitude"] < 0).any() and (truth["amplitude"] > 0).any()
    assert set(truth["kind"]) == {"spike", "box", "fred", "gaussian", "triangle"}
    assert random_anomalies(5, n, channels, seed=9) == random_anomalies(5, n, channels, seed=9)
    with pytest.raises(ValueError, match="could only place"):
        random_anomalies(10, 1000, channels, kinds=["box"], durations={"box": (200, 200)})


def test_match_events():
    truth = pd.DataFrame({"anomaly_id": [0, 1, 2], "start_index": [100, 500, 900], "stop_index": [120, 510, 950]})
    detections = pd.DataFrame({"event_id": [7, 8, 9], "start_index": [98, 300, 952],
                               "detection_index": [104, 305, 955], "stop_index": [130, 310, 960]})
    truth_out, det_out, summary = match_events(detections, truth)
    assert truth_out["detected"].tolist() == [True, False, False]
    assert truth_out["start_error"].iat[0] == -2 and truth_out["delay"].iat[0] == 4
    assert det_out["matched"].tolist() == [True, False, False]
    assert summary == {"n_anomalies": 3, "n_detected": 1, "efficiency": 1 / 3, "n_events": 3, "n_false": 2}
    truth_out, _, summary = match_events(detections, truth, tolerance=2)
    assert truth_out["detected"].tolist() == [True, False, True] and summary["n_false"] == 1
    with pytest.raises(ValueError, match="missing"):
        match_events(detections.drop(columns="stop_index"), truth)


def test_benchmark_end_to_end_with_true_background():
    df, cols = gaussian_dataset(2, 30_000 / 86_400, seed=5)
    resets = np.zeros(len(df), dtype=bool)
    resets[15_000] = True
    anomalies = random_anomalies(12, len(df), cols, resets=resets, kinds=["box", "fred"],
                                 z_range=(15.0, 30.0), durations={"box": (5, 100), "fred": (20, 200)},
                                 seed=2)
    data, truth = inject(df, anomalies)

    def n_events(frame):
        trig = Trigger(frame, cols, [f"{c}_bkg" for c in cols], thresholds=8.0)
        trig.run(reset_condition=resets)
        trig.identify_and_merge_triggers(merge_interval=60)
        return trig.get_detections_df()

    truth_out, _, summary = match_events(n_events(data), truth)
    assert summary["efficiency"] == 1.0
    # the events unrelated to the injections are the false alarms that the noise gives by itself
    assert summary["n_false"] == len(n_events(df))
    # a fred starts at 1% of its peak: its first samples are invisible, hence the relative bound
    assert (truth_out["start_error"].abs() <= np.maximum(10, 0.25 * truth_out["duration"])).all()
