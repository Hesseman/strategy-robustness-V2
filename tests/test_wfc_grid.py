import numpy as np
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.surface import metric_values, window_metrics
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db
from robustness.wfc_grid import WFCWindow, pearson, spearman, top_n_default, top_n_summary, wfc_bands, wfc_test, wfc_window
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


def _ww(x, y):
    """A WFCWindow carrying only x/y; the other fields are placeholders."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return WFCWindow(index=1, label="w", complete=True, n_points=int(np.isfinite(x).sum()), n_dropped=0, x=x, y=y,
                     spearman=0.0, pearson=0.0, n_pos_is=0, pos_oos_frac=0.0, p_value=1.0, null=np.empty(0), pick_index=0,
                     pick_is_pct=float("nan"), pick_oos_pct=float("nan"), quadrant="", insufficient=False)


def test_top_n_summary_finds_where_the_best_in_sample_landed():
    x = np.arange(1.0, 13.0)
    t = top_n_summary(_ww(x, x[::-1]), 4)                # the in-sample best are the out-of-sample worst
    assert t.n == 4 and t.indices == [11, 10, 9, 8] and t.n_valid == 12
    assert t.median_oos_rank == 10.5 and t.n_positive_oos == 4
    assert top_n_summary(_ww(x, x[::-1] - 6.5), 4).n_positive_oos == 0
    x2 = x.copy(); x2[0] = np.nan
    t2 = top_n_summary(_ww(x2, x))
    assert t2.n_valid == 11 and t2.n == top_n_default(11) == 3 and t2.indices == [11, 10, 9] and t2.median_oos_rank == 2.0
    assert top_n_default(240) == 10 and top_n_default(28) == 9 and top_n_default(2) == 1
    empty = top_n_summary(_ww([np.nan] * 3, [1.0, 2.0, 3.0]))
    assert empty.n == 0 and empty.indices == [] and np.isnan(empty.median_oos_rank)


def test_wfc_bands_cover_every_valid_point_once():
    rng = np.random.default_rng(3)
    x = rng.normal(size=27); x[5] = np.nan
    y = x * 2 + rng.normal(size=27)
    bands = wfc_bands(_ww(x, y), 10)
    assert [b.band for b in bands] == list(range(1, 11)) and sum(b.n for b in bands) == 26
    assert max(b.n for b in bands) - min(b.n for b in bands) <= 1
    assert bands[0].is_mean == max(b.is_mean for b in bands) and bands[0].oos_mean > bands[-1].oos_mean
    assert all(0.0 <= b.oos_pos_share <= 1.0 for b in bands)
    assert wfc_bands(_ww(x[:8], y[:8]), 10) == []