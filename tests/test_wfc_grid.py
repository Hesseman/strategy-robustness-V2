import numpy as np
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.surface import metric_values, window_metrics
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db
from robustness.wfc_grid import pearson, spearman, wfc_test, wfc_window
from robustness.windows import derive_windows

AX = {"A": list(range(6)), "B": list(range(6))}


def _run(structure, seed=5, metric="NP", n_days=1200):
    text, sched = make_multiwalk(AX, n_days=n_days, seed=seed, structure=structure, split=800)
    grid = parse_multiwalk_text(text)
    windows = derive_windows(parse_walkforward_db(make_walkforward_db(sched))[0], grid.dates)
    pairs = [(metric_values(window_metrics(grid, w.is_mask), metric), metric_values(window_metrics(grid, w.oos_mask), metric)) for w in windows]
    return wfc_test(pairs, windows, grid, metric=metric, n_null=199, seed=0)


def test_spearman_and_pearson_basics():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan]); y = np.array([2.0, 4.0, 6.0, 9.0, 1.0])
    assert spearman(x, y) == pytest.approx(1.0) and pearson(x, y) == pytest.approx(np.corrcoef(x[:4], y[:4])[0, 1])
    assert spearman(np.ones(5), np.arange(5.0)) == 0.0
    assert spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0   # fewer than 3 points


def test_persistent_surface_passes():
    r = _run("persistent")
    assert r.n_complete == 1 and r.windows[0].spearman > 0.5 and r.windows[0].p_value < 0.05
    assert r.pooled_p < 0.05 and r.pooled_pos_oos_frac >= 0.5 and r.passed and r.reasons == []
    assert r.windows[0].quadrant.startswith("structural edge")
    assert r.windows[0].pick_is_pct > 90.0


def test_noise_surface_fails_on_correlation():
    r = _run("noise")
    assert not r.passed and "wfc_low_corr" in r.reasons


def test_decayed_surface_fails_on_oos_sign():
    r = _run("decay")
    assert not r.passed and "wfc_no_oos_edge" in r.reasons
    assert r.windows[0].pos_oos_frac < 0.5


def test_null_is_torus_shift_on_full_grid_and_deterministic():
    r1, r2 = _run("persistent", seed=7), _run("persistent", seed=7)
    w = r1.windows[0]
    assert w.null.size == 35                       # every non-zero shift of a 6x6 grid
    assert r1.windows[0].p_value == r2.windows[0].p_value and r1.pooled_p == r2.pooled_p
    assert 1 / 36 <= w.p_value <= 1.0


def test_incomplete_or_tiny_window_is_insufficient():
    text, sched = make_multiwalk({"A": [1, 2], "B": [1, 2]}, n_days=300, seed=1, structure="persistent", split=200)
    grid = parse_multiwalk_text(text)
    windows = derive_windows(parse_walkforward_db(make_walkforward_db(sched))[0], grid.dates)
    x = metric_values(window_metrics(grid, windows[0].is_mask), "NP"); y = metric_values(window_metrics(grid, windows[0].oos_mask), "NP")
    r = wfc_test([(x, y)], windows, grid, metric="NP", n_null=50, seed=0)
    assert r.windows[0].insufficient and not r.passed and r.reasons == ["wfc_insufficient"]


def test_window_null_correlates_over_all_pairs_finite_after_the_shift():
    """Regression: the per-window null must use the raw x (like the pooled null), not x with the
    original y's holes copied in - otherwise dropped OOS cells shrink every null draw's point set."""
    from robustness.wfc_grid import _Shifter
    text, sched = make_multiwalk(AX, n_days=1200, seed=5, structure="persistent", split=800)
    grid = parse_multiwalk_text(text)
    windows = derive_windows(parse_walkforward_db(make_walkforward_db(sched))[0], grid.dates)
    x = metric_values(window_metrics(grid, windows[0].is_mask), "NP")
    y = metric_values(window_metrics(grid, windows[0].oos_mask), "NP").copy()
    y[:12] = np.nan                                  # 12 combinations dropped on the OOS side only
    x[20:23] = np.nan   # 3 IS-side holes where the OOS side is finite: the two null constructions differ here
    sh = _Shifter(grid)
    w = wfc_window(x, y, windows[0], sh, n_null=199, seed=0)
    assert w.n_points == 21 and w.n_dropped == 15 and w.null.size == 35
    rng = np.random.default_rng(0 + 1000 * windows[0].index)
    yn = np.where(np.isfinite(y), y, np.nan)
    expected = np.array([spearman(x, sh.apply(yn, s, rng)) for s in sh.all_shifts(rng, 199)])
    assert np.allclose(w.null, expected)
