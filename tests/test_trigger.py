import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tslies.stats import gaussian_focus
from tslies.trigger import Trigger

REPO = Path(__file__).resolve().parents[1]


def make_df(n=2000, channels=("a",), seed=0, freq="10s"):
    """Noise around a varying background with varying std, sampled every 10 s (not 1 s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    data = {"datetime": pd.date_range("2024-03-01", periods=n, freq=freq, tz="UTC")}
    for i, c in enumerate(channels):
        pred = 100 + 10 * np.sin(2 * np.pi * t / 500 + i)
        std = 2 + np.cos(2 * np.pi * t / 700)
        data[c] = pred + std * rng.standard_normal(n)
        data[f"{c}_pred"] = pred
        data[f"{c}_std"] = std
    return pd.DataFrame(data)


def add_z(df, channel, start, stop, amplitude):
    """Add ``amplitude`` noise standard deviations to ``channel`` on rows [start, stop)."""
    df.loc[start:stop - 1, channel] += amplitude * df.loc[start:stop - 1, f"{channel}_std"]


def events_of(df, channels=("a",), **kwargs):
    merge = kwargs.pop("merge_interval", 3)
    reset = kwargs.pop("reset_condition", None)
    trig = Trigger(df, list(channels), [f"{c}_pred" for c in channels], **kwargs)
    trig.run(reset_condition=reset)
    trig.identify_and_merge_triggers(merge_interval=merge)
    return trig, trig.get_detections_df()


def test_import_has_no_side_effects(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "TSLIES_DIR"}
    env["PYTHONPATH"] = str(REPO)
    subprocess.run([sys.executable, "-c", "import tslies.trigger"], cwd=tmp_path, env=env, check=True)
    assert list(tmp_path.iterdir()) == []


def test_focus_significance_matches_stats_core():
    df = make_df()
    add_z(df, "a", 500, 540, 1.5)
    trig = Trigger(df, ["a"], ["a_pred"], thresholds=7.0)
    trig.run()
    z = (df["a"] - df["a_pred"]) / df["a_std"]
    np.testing.assert_allclose(trig.results["a_significance"], gaussian_focus(z, restart_above=7.0).significance)


def test_background_after_alarm_matches_stats_core_and_covers_a_long_anomaly():
    df = make_df()
    add_z(df, "a", 500, 700, 1.0)
    trig, det = events_of(df, thresholds=6.0, after_alarm="background", merge_interval=60)
    z = (df["a"] - df["a_pred"]) / df["a_std"]
    np.testing.assert_allclose(trig.results["a_significance"], gaussian_focus(z, background_above=6.0).significance)
    assert len(det) == 1
    # with restarts the same anomaly raises fewer alarms, so fewer of its samples are marked
    restarted, _ = events_of(df, thresholds=6.0, merge_interval=60)
    assert (trig.results["a_significance"] > 6.0).sum() > (restarted.results["a_significance"] > 6.0).sum()


def test_long_anomaly_gives_repeated_alarms_in_one_event_and_no_tail():
    df = make_df(n=3000)
    add_z(df, "a", 1000, 1300, 1.0)  # strength 17: several alarms while it lasts
    trig = Trigger(df, ["a"], ["a_pred"], thresholds=5.5)
    trig.run()
    alarms = np.flatnonzero(trig.results["a_significance"].to_numpy() > 5.5)
    assert alarms.size >= 3 and alarms.min() >= 1000 and alarms.max() < 1300
    assert not trig.mask[1300:].any()  # nothing left after the end
    trig.identify_and_merge_triggers(merge_interval=60)
    det = trig.get_detections_df()
    assert len(det) == 1
    assert abs(det["start_index"].iat[0] - 1000) <= 10 and 1200 <= det["stop_index"].iat[0] < 1300
    trig.identify_and_merge_triggers(merge_interval=10)
    assert len(trig.get_detections_df()) > 1  # too short to bridge the gaps between alarms


def test_single_burst_gives_one_event_with_its_changepoint():
    df = make_df()
    add_z(df, "a", 500, 540, 1.5)
    _, det = events_of(df, thresholds=7.0)
    assert len(det) == 1
    row = det.iloc[0]
    assert abs(row["start_index"] - 500) <= 3
    assert 500 <= row["detection_index"] <= row["peak_index"] <= 545
    assert row["triggered_faces"] == "a"
    assert row["start_datetime"] == df["datetime"].iat[row["start_index"]]
    assert row["stop_index"] >= row["peak_index"]


def test_no_events_on_pure_noise():
    df = make_df(n=5000, channels=("a", "b", "c"))
    trig, det = events_of(df, channels=("a", "b", "c"), thresholds=7.0)
    assert trig.merged_anomalies == {}
    assert det.empty and "start_datetime" in det.columns


def test_events_on_first_and_last_sample():
    df = make_df(n=300)
    add_z(df, "a", 0, 1, 9.0)
    add_z(df, "a", 299, 300, 9.0)
    _, det = events_of(df, trigger_type="z_score", thresholds=6.0)
    assert det["start_index"].tolist() == [0, 299]
    assert det["stop_index"].tolist() == [0, 299]


def test_mu_min_ignores_slow_bias_but_keeps_bursts():
    df = make_df(n=20_000, seed=1)
    add_z(df, "a", 0, 20_000, 0.2)
    trig = Trigger(df, ["a"], ["a_pred"], thresholds=7.0)
    assert trig.run()["anomaly"].sum() > 1000
    add_z(df, "a", 8000, 8050, 3.0)
    _, det = events_of(df, thresholds=7.0, mu_min=0.5)
    assert len(det) == 1 and abs(det["start_index"].iat[0] - 8000) <= 3


def test_coincidence_between_groups():
    channels = ("a1", "a2", "b1", "c1")
    groups = {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1"]}
    df = make_df(channels=channels)
    add_z(df, "a1", 300, 330, 2.0)
    add_z(df, "a2", 300, 330, 2.0)
    _, det = events_of(df, channels=channels, thresholds=7.0, groups=groups, min_groups=2)
    assert det.empty  # two channels, but one group
    _, det = events_of(df, channels=channels, thresholds=7.0)
    assert len(det) == 1  # plain OR
    add_z(df, "b1", 1200, 1230, 2.0)
    add_z(df, "c1", 1200, 1230, 2.0)
    _, det = events_of(df, channels=channels, thresholds=7.0, groups=groups, min_groups=2)
    assert len(det) == 1 and set(det["triggered_faces"].iat[0].split("/")) == {"b1", "c1"}


def test_non_finite_residuals_are_gaps_that_restart_the_detector():
    df = make_df(n=400)
    add_z(df, "a", 150, 200, 3.0)
    df.loc[200:209, "a_std"] = 0.0  # a broken std: residuals are inf or nan
    trig = Trigger(df, ["a"], ["a_pred"], thresholds=7.0)
    trig.run()
    sig = trig.results["a_significance"]
    assert sig.iloc[200:210].isna().all()
    assert trig.results["a_length"].iat[210] <= 1  # restarted, not carrying the burst
    assert not trig.results["a_triggered"].iloc[200:210].any()


def test_resets_split_events_and_restart_detector():
    df = make_df(n=1000)
    add_z(df, "a", 400, 600, 1.5)
    reset = np.zeros(1000, dtype=bool)
    reset[500] = True
    trig, det = events_of(df, thresholds=7.0, reset_condition=reset, merge_interval=1000)
    assert len(det) == 2
    assert det["start_index"].iat[1] >= 500
    assert det["stop_index"].iat[0] < 500


def test_strong_burst_leaves_no_tail_that_hides_the_next_one():
    df = make_df(n=6000)
    add_z(df, "a", 1000, 1100, 3.0)  # z = 30: without restarts it would stay over 7 for ~1800 samples
    add_z(df, "a", 2000, 2030, 3.0)
    trig = Trigger(df, ["a"], ["a_pred"], thresholds=7.0)
    trig.run()
    assert not trig.mask[1100:2000].any()
    trig.identify_and_merge_triggers(merge_interval=10)
    det = trig.get_detections_df()
    assert len(det) == 2
    assert abs(det["start_index"].iat[0] - 1000) <= 3 and abs(det["start_index"].iat[1] - 2000) <= 3
    assert det["stop_index"].iat[0] < 1100 and det["stop_index"].iat[1] < 2030


def test_merge_interval_in_samples():
    df = make_df(n=500)
    add_z(df, "a", 100, 101, 9.0)
    add_z(df, "a", 111, 112, 9.0)  # 10 quiet samples in between
    assert len(events_of(df, trigger_type="z_score", thresholds=6.0, merge_interval=9)[1]) == 2
    assert len(events_of(df, trigger_type="z_score", thresholds=6.0, merge_interval=10)[1]) == 1


def test_poisson_focus_on_counts():
    rng = np.random.default_rng(3)
    n = 3000
    lam = 5 + 3 * np.sin(np.arange(n) / 200)
    factor = np.ones(n)
    factor[1000:1030] = 3.0
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC"),
                       "k": rng.poisson(lam * factor).astype(float), "k_pred": lam})
    trig = Trigger(df, ["k"], ["k_pred"], trigger_type="poisson_focus", thresholds=7.0)
    trig.run()
    trig.identify_and_merge_triggers()
    det = trig.get_detections_df()
    assert len(det) == 1 and abs(det["start_index"].iat[0] - 1000) <= 5
    df["k"] = df["k"] / 7.3  # rates, not counts
    with pytest.raises(ValueError, match="counts"):
        Trigger(df, ["k"], ["k_pred"], trigger_type="poisson_focus").run()


@pytest.mark.parametrize("trigger_type", ["focus", "z_score"])
def test_sides(trigger_type):
    df = make_df(n=600)
    add_z(df, "a", 300, 330, -2.5)
    thr = 7.0 if trigger_type == "focus" else 5.0
    assert events_of(df, trigger_type=trigger_type, thresholds=thr, side="up")[1].empty
    for side in ("down", "both"):
        trig, det = events_of(df, trigger_type=trigger_type, thresholds=thr, side=side)
        assert len(det) >= 1
    assert (trig.results.loc[trig.results["a_triggered"], "a_sign"] == -1).all()


def test_filter_from_catalog_matches_overlaps_including_containment():
    df = make_df(n=2000)
    add_z(df, "a", 500, 600, 1.5)
    trig, det = events_of(df, thresholds=7.0)
    start, stop = det["start_datetime"].iat[0], det["stop_datetime"].iat[0]
    naive = lambda ts: ts.tz_convert("UTC").tz_localize(None).isoformat()
    catalog = pd.DataFrame({
        "NAME": ["inside", "before"],
        "TIME": [naive(start + pd.Timedelta("100s")), "2020-01-01T00:00:00"],
        "END_TIME": [naive(stop - pd.Timedelta("100s")), "2020-01-01T00:01:00"],
    })
    before = {k: dict(v) for k, v in trig.merged_anomalies[0].items()}
    matched, results = trig.filter_from_catalog(catalog, trig.merged_anomalies, det)
    assert len(matched) == 1
    assert [c["NAME"] for c in matched["catalog_triggers"].iat[0]] == ["inside"]
    assert results[0]["a"]["catalog_triggers"][0]["NAME"] == "inside"
    assert trig.merged_anomalies[0] == before  # not modified in place
    empty, none = trig.filter_from_catalog(catalog.iloc[[1]], trig.merged_anomalies, det)
    assert empty.empty and none == {}


def test_index_and_input_are_not_assumed_or_modified():
    df = make_df()
    add_z(df, "a", 500, 540, 1.5)
    shifted = df.set_index(pd.Index(np.arange(len(df)) * 7 + 13))
    snapshot = shifted.copy()
    _, det_range = events_of(df, thresholds=7.0)
    _, det_shift = events_of(shifted, thresholds=7.0)
    pd.testing.assert_frame_equal(det_range, det_shift)
    pd.testing.assert_frame_equal(shifted, snapshot)


def test_save_detections_csv(tmp_path):
    df = make_df()
    add_z(df, "a", 500, 540, 1.5)
    trig, det = events_of(df, thresholds=7.0)
    path = trig.save_detections_csv(det, file="x", suffix="_y", folder=tmp_path)
    assert Path(path).name == "detections_x_y.csv"
    assert len(pd.read_csv(path)) == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(y_cols=["a"], y_cols_pred=["a_pred", "a_std"]), "one column per channel"),
        (dict(y_cols=["zz"], y_cols_pred=["a_pred"]), "missing the columns"),
        (dict(thresholds={"b": 3.0}), "missing the channels"),
        (dict(thresholds=-1.0), "positive"),
        (dict(trigger_type="cusum"), "trigger_type"),
        (dict(trigger_type="poisson_focus", side="both"), "side='up'"),
        (dict(trigger_type="z_score", mu_min=1.0), "mu_min"),
        (dict(mu_min=-1.0), "mu_min"),
        (dict(after_alarm="forget"), "after_alarm"),
        (dict(trigger_type="z_score", after_alarm="background"), "after_alarm"),
        (dict(groups={"g": ["a", "b"]}), "partition"),
        (dict(min_groups=2), "min_groups"),
        (dict(time_col="when"), "missing the columns"),
    ],
)
def test_validation(kwargs, match):
    df = make_df()
    args = dict(y_cols=["a"], y_cols_pred=["a_pred"])
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Trigger(df, **args).run()


def test_call_order_and_reset_length():
    trig = Trigger(make_df(), ["a"], ["a_pred"])
    with pytest.raises(RuntimeError, match="run"):
        trig.identify_and_merge_triggers()
    with pytest.raises(ValueError, match="reset_condition"):
        trig.run(reset_condition=[True, False])


def test_plot_anomalies_end_to_end(tmp_path):
    script = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
sys.path.insert(0, {str(REPO / 'tests')!r})
from test_trigger import make_df, add_z
from tslies.trigger import Trigger
df = make_df()
add_z(df, 'a', 500, 540, 1.5)
trig = Trigger(df, ['a'], ['a_pred'], thresholds=7.0)
trig.run()
trig.identify_and_merge_triggers()
trig.plot_anomalies()
"""
    env = dict(os.environ, TSLIES_DIR=str(tmp_path), MPLBACKEND="Agg")
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True,
                   capture_output=True)
    assert len(list(tmp_path.glob("results/*/anomalies/*/plots/*.png"))) == 1
