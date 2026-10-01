import numpy as np
import pytest

from tslies.stats import GaussianFocus, PoissonFocus, gaussian_focus, poisson_focus


def brute_gaussian(x, mu_min=0.0):
    """LLR_t = max over segments ending at t and mu >= mu_min of mu S - n mu**2 / 2, by definition."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    llr = np.zeros(len(x))
    for t in range(1, len(x) + 1):
        s = c[t] - c[:t]
        n = t - np.arange(t)
        mu = np.maximum(s / n, mu_min)
        llr[t - 1] = max(0.0, np.max(mu * s - n * mu * mu / 2))
    return llr


def brute_poisson(counts, expected, mu_min=1.0):
    """LLR_t = max over segments ending at t and mu >= mu_min of K log(mu) - L (mu - 1)."""
    ck = np.concatenate([[0.0], np.cumsum(counts)])
    cl = np.concatenate([[0.0], np.cumsum(expected)])
    llr = np.zeros(len(counts))
    for t in range(1, len(counts) + 1):
        k = ck[t] - ck[:t]
        lam = cl[t] - cl[:t]
        mu = np.maximum(k / lam, mu_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            seg = np.where(k > 0, k * np.log(mu), 0.0) - lam * (mu - 1)
        llr[t - 1] = max(0.0, np.max(seg))
    return llr


def gaussian_sequences():
    rng = np.random.default_rng(1)
    yield "white", rng.standard_normal(400)
    burst = rng.standard_normal(400)
    burst[100:130] += 1.5
    burst[250:253] += 4.0
    yield "bursts", burst
    yield "negative_drift", rng.standard_normal(400) - 0.3
    yield "positive_drift", rng.standard_normal(400) + 0.1
    yield "integer_ties", rng.integers(-1, 3, 400).astype(float)
    yield "zeros_then_step", np.r_[np.zeros(50), np.ones(30), np.zeros(50)]


def poisson_sequences():
    rng = np.random.default_rng(2)
    lam = rng.uniform(0.5, 20.0, 400)
    yield "varying_rate", rng.poisson(lam).astype(float), lam
    factor = np.ones(400)
    factor[150:190] = 1.8
    yield "burst", rng.poisson(lam * factor).astype(float), lam
    low = np.full(400, 0.05)
    yield "low_rate_ties", rng.poisson(low).astype(float), low
    yield "deficit", rng.poisson(lam * 0.7).astype(float), lam


@pytest.mark.parametrize("name,x", list(gaussian_sequences()))
def test_gaussian_matches_brute_force(name, x):
    res = gaussian_focus(x)
    np.testing.assert_allclose(res.llr, brute_gaussian(x), rtol=1e-9, atol=1e-12)
    # the reported change must be a segment that attains the maximum
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(1, len(x) + 1)
    has = res.length > 0
    s = c[t[has]] - c[t[has] - res.length[has]]
    np.testing.assert_allclose(s * s / (2 * res.length[has]), res.llr[has], rtol=1e-9)
    assert np.all(res.llr[~has] == 0)


@pytest.mark.parametrize("name,counts,expected", list(poisson_sequences()))
def test_poisson_matches_brute_force(name, counts, expected):
    res = poisson_focus(counts, expected)
    np.testing.assert_allclose(res.llr, brute_poisson(counts, expected), rtol=1e-9, atol=1e-12)
    ck = np.concatenate([[0.0], np.cumsum(counts)])
    cl = np.concatenate([[0.0], np.cumsum(expected)])
    t = np.arange(1, len(counts) + 1)
    has = res.length > 0
    k = ck[t[has]] - ck[t[has] - res.length[has]]
    lam = cl[t[has]] - cl[t[has] - res.length[has]]
    np.testing.assert_allclose(k * np.log(k / lam) - (k - lam), res.llr[has], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mu_min", [0.3, 1.0, 2.5])
@pytest.mark.parametrize("name,x", list(gaussian_sequences()))
def test_gaussian_mu_min_matches_brute_force(name, x, mu_min):
    res = gaussian_focus(x, mu_min=mu_min)
    np.testing.assert_allclose(res.llr, brute_gaussian(x, mu_min), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("mu_min", [1.1, 1.5, 3.0])
@pytest.mark.parametrize("name,counts,expected", list(poisson_sequences()))
def test_poisson_mu_min_matches_brute_force(name, counts, expected, mu_min):
    res = poisson_focus(counts, expected, mu_min=mu_min)
    np.testing.assert_allclose(res.llr, brute_poisson(counts, expected, mu_min), rtol=1e-9, atol=1e-12)


def test_mu_min_ignores_small_persistent_bias():
    # a bias of 0.2 sigma lasting 20000 samples: z = 0.2 * sqrt(20000) ~ 28 for standard FOCuS
    x = np.full(20_000, 0.2)
    assert gaussian_focus(x).significance[-1] == pytest.approx(0.2 * np.sqrt(20_000))
    assert np.all(gaussian_focus(x, mu_min=0.5).llr == 0)
    # while a shift above mu_min is still detected
    x[5000:5100] += 3.0
    assert gaussian_focus(x, mu_min=0.5).significance.max() > 25


def restart_by_hand(detector, pairs, threshold):
    """Reference: update one sample at a time and start again after every alarm."""
    out = []
    for args in pairs:
        llr, length = detector.update(*args)
        out.append((llr, length))
        if np.sqrt(2 * llr) > threshold:
            detector.reset()
    return np.array(out).T


@pytest.mark.parametrize("threshold", [2.0, 3.5, 5.0])
@pytest.mark.parametrize("name,x", list(gaussian_sequences()))
def test_gaussian_restart_matches_step_by_step(name, x, threshold):
    res = gaussian_focus(x, restart_above=threshold, mu_min=0.2)
    llr, length = restart_by_hand(GaussianFocus(0.2), [(v,) for v in x], threshold)
    np.testing.assert_allclose(res.llr, llr, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(res.length, length)


@pytest.mark.parametrize("threshold", [2.0, 4.0])
@pytest.mark.parametrize("name,counts,expected", list(poisson_sequences()))
def test_poisson_restart_matches_step_by_step(name, counts, expected, threshold):
    res = poisson_focus(counts, expected, restart_above=threshold)
    llr, length = restart_by_hand(PoissonFocus(), list(zip(counts, expected)), threshold)
    np.testing.assert_allclose(res.llr, llr, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(res.length, length)


def test_restart_forgets_a_change_after_its_alarms():
    x = np.r_[np.zeros(100), np.ones(300), np.zeros(3000)]  # +1 sigma for 300 samples
    kept = gaussian_focus(x).significance
    restarted = gaussian_focus(x, restart_above=5.0).significance
    assert kept[399] == pytest.approx(np.sqrt(300)) and kept[399 + 2000] > 5  # never forgotten
    alarms = np.flatnonzero(restarted > 5.0)
    assert alarms.size > 5 and alarms.min() >= 100 and alarms.max() < 400  # repeated, only while it lasts
    assert np.all(restarted[400:] < 5.0)


@pytest.mark.parametrize("side", ["up", "down", "both"])
def test_restart_each_side_on_its_own(side):
    x = np.random.default_rng(9).standard_normal(2000)
    x[500:600] += 1.5
    x[1200:1300] -= 1.5
    res = gaussian_focus(x, side=side, restart_above=4.0)
    up, down = gaussian_focus(x, restart_above=4.0), gaussian_focus(-x, restart_above=4.0)
    expected = {"up": up.llr, "down": down.llr, "both": np.maximum(up.llr, down.llr)}[side]
    np.testing.assert_allclose(res.llr, expected)


def test_restart_does_not_change_quiet_data():
    x = np.random.default_rng(10).standard_normal(5000)
    np.testing.assert_array_equal(gaussian_focus(x, restart_above=50.0).llr, gaussian_focus(x).llr)


def background_by_hand(make_detector, pairs, threshold, background):
    """Reference: run again from the start at every sample, with the alarmed samples replaced by background."""
    seen, out = [], []
    for args in pairs:
        detector = make_detector()
        for previous in seen:
            detector.update(*previous)
        llr, length = detector.update(*args)
        out.append((llr, length))
        # the library's test: significance > threshold, as LLR > threshold**2 / 2 (no rounding on ties)
        seen.append(background(args) if llr > threshold ** 2 / 2 else args)
    return np.array(out).T


@pytest.mark.parametrize("threshold", [2.0, 3.5, 5.0])
@pytest.mark.parametrize("name,x", list(gaussian_sequences()))
def test_gaussian_background_after_alarm_matches_recomputing(name, x, threshold):
    res = gaussian_focus(x, background_above=threshold, mu_min=0.2)
    llr, length = background_by_hand(lambda: GaussianFocus(0.2), [(v,) for v in x], threshold, lambda args: (0.0,))
    np.testing.assert_allclose(res.llr, llr, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(res.length, length)


@pytest.mark.parametrize("threshold", [2.0, 4.0])
@pytest.mark.parametrize("name,counts,expected", list(poisson_sequences()))
def test_poisson_background_after_alarm_matches_recomputing(name, counts, expected, threshold):
    res = poisson_focus(counts, expected, background_above=threshold)
    llr, length = background_by_hand(PoissonFocus, list(zip(counts, expected)), threshold,
                                     lambda args: (args[1], args[1]))  # as many counts as expected
    np.testing.assert_allclose(res.llr, llr, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(res.length, length)


def test_background_after_alarm_keeps_the_evidence_just_below_the_threshold():
    x = np.r_[np.zeros(100), np.ones(300), np.zeros(3000)]  # +1 sigma for 300 samples
    kept = gaussian_focus(x, background_above=5.0).significance
    restarted = gaussian_focus(x, restart_above=5.0).significance
    alarms = np.flatnonzero(kept > 5.0)
    assert alarms.size > 100 and alarms.min() >= 100 and alarms.max() < 400  # many, only while it lasts
    assert kept.max() < 5.2  # never much above the threshold
    assert kept[499] > 4.0 > restarted[499]  # 100 samples after the end: remembered, unlike with restarts


def test_background_after_alarm_does_not_change_quiet_data():
    x = np.random.default_rng(10).standard_normal(5000)
    np.testing.assert_array_equal(gaussian_focus(x, background_above=50.0).llr, gaussian_focus(x).llr)


def test_single_spike_significance_is_its_z_score():
    res = gaussian_focus([0.0, 0.0, 3.0, 0.0])
    assert res.significance[2] == pytest.approx(3.0)
    assert res.length[2] == 1


def test_step_significance_is_segment_z_score():
    x = np.r_[np.zeros(20), np.ones(25), np.zeros(5)]
    res = gaussian_focus(x)
    assert res.significance[44] == pytest.approx(5.0)
    assert res.length[44] == 25


def test_resets_split_the_series():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(300) + 0.2
    resets = np.zeros(300, dtype=bool)
    resets[[100, 220]] = True
    res = gaussian_focus(x, resets=resets)
    for a, b in [(0, 100), (100, 220), (220, 300)]:
        np.testing.assert_allclose(res.llr[a:b], brute_gaussian(x[a:b]), rtol=1e-9, atol=1e-12)
    assert res.length[100] <= 1


def test_sides():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(300)
    up, down, both = (gaussian_focus(x, side=s) for s in ("up", "down", "both"))
    np.testing.assert_allclose(down.llr, gaussian_focus(-x).llr)
    np.testing.assert_allclose(both.llr, np.maximum(up.llr, down.llr))
    assert set(np.unique(both.sign)) <= {-1, 0, 1}
    assert np.all(both.sign[down.llr > up.llr] == -1)


def test_online_matches_batch():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(200)
    det = GaussianFocus()
    online = np.array([det.update(v)[0] for v in x])
    np.testing.assert_allclose(online, gaussian_focus(x).llr)

    lam = rng.uniform(1, 5, 200)
    k = rng.poisson(lam).astype(float)
    det = PoissonFocus()
    online = np.array([det.update(a, b)[0] for a, b in zip(k, lam)])
    np.testing.assert_allclose(online, poisson_focus(k, lam).llr)


def test_significance_never_below_single_sample_z():
    x = np.random.default_rng(6).standard_normal(1000)
    assert np.all(gaussian_focus(x).significance >= np.maximum(x, 0) - 1e-12)


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: gaussian_focus([0.0, np.nan]), "non-finite"),
        (lambda: gaussian_focus(np.zeros((2, 2))), "one-dimensional"),
        (lambda: gaussian_focus([0.0, 1.0], resets=[True]), "resets"),
        (lambda: gaussian_focus([0.0], side="left"), "side"),
        (lambda: poisson_focus([1.0, -1.0], [1.0, 1.0]), "non-negative"),
        (lambda: poisson_focus([0.5, 1.0], [1.0, 1.0]), "not integers"),
        (lambda: poisson_focus([1.0, 1.0], [1.0, 0.0]), "strictly positive"),
        (lambda: poisson_focus([1.0], [1.0, 1.0]), "differ in length"),
        (lambda: gaussian_focus([0.0], mu_min=-1.0), "mu_min"),
        (lambda: gaussian_focus([0.0], mu_min=np.inf), "mu_min"),
        (lambda: poisson_focus([1.0], [1.0], mu_min=0.5), "mu_min"),
        (lambda: gaussian_focus([0.0, 1.0], restart_above=0.0), "restart_above"),
        (lambda: gaussian_focus([0.0, 1.0], restart_above=np.nan), "restart_above"),
        (lambda: poisson_focus([1.0], [1.0], restart_above=-2.0), "restart_above"),
        (lambda: gaussian_focus([0.0, 1.0], background_above=0.0), "background_above"),
        (lambda: poisson_focus([1.0], [1.0], background_above=np.inf), "background_above"),
        (lambda: gaussian_focus([0.0], restart_above=3.0, background_above=3.0), "either"),
        (lambda: poisson_focus([1.0], [1.0], restart_above=3.0, background_above=3.0), "either"),
    ],
)
def test_input_validation(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_non_integer_counts_can_be_allowed():
    res = poisson_focus([0.5, 4.5], [1.0, 1.0], require_integer_counts=False)
    np.testing.assert_allclose(res.llr, brute_poisson(np.array([0.5, 4.5]), np.ones(2)))
