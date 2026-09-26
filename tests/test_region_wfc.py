"""Region lift on the optimisation grid (docs/wfc-region-lift.md):
the in-sample top 20% of the pooled in-sample surface, judged by its out-of-sample net profit
against the grid average, under the block sign-flip null."""
import math

import numpy as np
import pandas as pd
import pytest

from robustness.multiwalk_text import MultiWalkGrid
from robustness.plateau_grid import neighbours
from robustness.region_wfc import largest_component, neighbourhoods, pool, region_test, ridge, top_set
from robustness.windows import Window

# 3x3 grid, first axis fastest (MultiWalk's grid-row order): index i sits at (i % 3, i // 3).
_POS_3X3 = np.array([[a, b] for b in range(3) for a in range(3)])


def _grid_pos(shape):
    """Grid positions, first axis fastest."""
    import itertools
    return np.array([tuple(reversed(c)) for c in itertools.product(*[range(s) for s in reversed(shape)])], dtype=int)


def test_neighbourhood_is_the_cell_itself_plus_its_chebyshev_one_neighbours_without_wrap():
    for pos in (_POS_3X3, _grid_pos((3, 2, 2))):
        nb = neighbourhoods(pos)
        assert len(nb) == len(pos)
        for i, idx in enumerate(nb):
            assert idx[0] == i and sorted(idx[1:].tolist()) == sorted(neighbours(pos, i).tolist())
    assert sorted(neighbourhoods(_POS_3X3)[0].tolist()) == [0, 1, 3, 4]          # a corner does not wrap


def test_pool_is_the_equal_weight_mean_of_each_neighbourhood():
    v = np.arange(9.0)                                                          # value = index
    expected = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]                    # hand-traced, e.g. corner 0: mean(0, 1, 3, 4)
    nb = neighbourhoods(_POS_3X3)
    assert pool(v, nb).tolist() == expected
    assert pool(np.stack([v, 2 * v]), nb).tolist() == [expected, [2 * e for e in expected]]


def test_top_set_breaks_ties_in_grid_order():
    assert top_set(np.array([1.0, 3.0, 3.0, 2.0]), 2).tolist() == [1, 2]
    assert top_set(np.array([5.0, 5.0, 5.0]), 2).tolist() == [0, 1]
    assert top_set(np.array([0.0, 9.0, 4.0, 7.0]), 3).tolist() == [1, 3, 2]


def test_largest_component_connects_through_diagonals_and_prefers_the_lowest_index_on_ties():
    pos = _grid_pos((4, 4))
    at = {tuple(p): i for i, p in enumerate(pos.tolist())}
    mask = np.zeros(16, dtype=bool)
    mask[[at[(0, 0)], at[(1, 1)]]] = True                                       # diagonal pair: one component
    mask[[at[(3, 0)], at[(3, 1)], at[(3, 2)]]] = True                           # three in a column
    got = largest_component(mask, pos)
    assert set(np.flatnonzero(got).tolist()) == {at[(3, 0)], at[(3, 1)], at[(3, 2)]}
    mask[at[(3, 2)]] = False                                                    # now two components of two
    assert set(np.flatnonzero(largest_component(mask, pos)).tolist()) == {at[(0, 0)], at[(1, 1)]}
    assert not largest_component(np.zeros(16, dtype=bool), pos).any()


def test_ridge_is_the_largest_connected_top_fifth():
    pos = _grid_pos((5, 5))
    at = {tuple(p): i for i, p in enumerate(pos.tolist())}
    v = np.zeros(25)
    plus = [at[(2, 2)], at[(1, 2)], at[(3, 2)], at[(2, 1)], at[(2, 3)]]
    v[plus] = [9.0, 8.0, 8.0, 7.0, 7.0]
    v[at[(0, 4)]] = 1.0                                                         # below the top fifth
    assert set(np.flatnonzero(ridge(v, pos)).tolist()) == set(plus)


def _line_grid(is_np, oos_rows, grid_row=1):
    """A one-parameter grid of n cells in a line: two in-sample days (the whole in-sample net
    profit on the first) followed by the given out-of-sample daily rows; one complete window."""
    is_np = np.asarray(is_np, dtype=float)
    daily = np.column_stack([is_np, np.zeros_like(is_np), np.asarray(oos_rows, dtype=float)])
    n, t = daily.shape
    dates = pd.bdate_range("2021-01-04", periods=t)
    pos = np.arange(n)[:, None]
    grid = MultiWalkGrid(param_names=["A"], params=pos.astype(float), axes=[np.arange(n, dtype=float)], grid_pos=pos,
                         dates=dates, daily_pnl=daily, closed_pnl=daily.copy(),
                         exit_dates=[dates.values[daily[i] != 0].astype("datetime64[D]") for i in range(n)],
                         exit_pnl=[daily[i][daily[i] != 0] for i in range(n)])
    is_mask = np.arange(t) < 2
    w = Window(index=1, label="w", is_start=dates[0], is_end=dates[1], oos_start=dates[2], oos_end_nominal=dates[-1],
               oos_end=dates[-1], is_mask=is_mask, oos_mask=~is_mask, complete=True, grid_row=grid_row, params=(0.0,))
    return grid, w


# Hand trace, 10 cells, region size k = round(0.2 x 10) = 2.
#   IS net profit       x  = [0, 1, 2, 3, 4, 10, 4, 3, 2, 9]            -> the raw top 2 would be [5, 9]
#   pooled IS           x~ = [0.5, 1, 2, 3, 17/3, 6, 17/3, 3, 14/3, 5.5] -> R = [5, 4] (tie 4/6 -> grid order)
#   OOS net profit      y  = [2, 2, 2, 2, 8, 6, 2, 2, 2, 2]: grid mean 3, R mean 7, SD sqrt(4.2)
#   pooled OOS          y~ = [2, 2, 2, 4, 16/3, 16/3, 10/3, 2, 2, 2]    -> top 2 = [4, 5]: precision 1
#   ridges (top-fifth quantile, linear): IS {4, 5, 6} (threshold 17/3), OOS {4, 5} (threshold 4.27)
#   picks (share of cells with lower OOS): best IS cell 5 -> 80, best pooled 5 -> 80, R mean 7 -> 90, cell 0 -> 0
_X = [0, 1, 2, 3, 4, 10, 4, 3, 2, 9]
_Y = [2, 2, 2, 2, 8, 6, 2, 2, 2, 2]
_OOS_DISTINCT = [[i, y - i] for i, y in enumerate(_Y)]        # same net profit, ten different daily patterns
_OOS_TIED = [[y, 0] for y in _Y]                               # eight identical variants: 3 patterns < 2k


def test_window_statistics_hand_traced():
    grid, w = _line_grid(_X, _OOS_DISTINCT)
    r = region_test(grid, [w], n_null=20, seed=0).windows[0]
    assert r.k == 2 and r.region.tolist() == [5, 4]
    assert r.x.tolist() == _X and r.y.tolist() == _Y
    assert r.x_pooled.tolist() == pytest.approx([0.5, 1, 2, 3, 17 / 3, 6, 17 / 3, 3, 14 / 3, 5.5])
    assert r.grid_oos_mean == pytest.approx(3.0) and r.region_oos_mean == pytest.approx(7.0)
    assert r.oos_sd == pytest.approx(math.sqrt(4.2)) and r.lift == pytest.approx(4.0 / math.sqrt(4.2))
    assert r.oos_positive_share == 1.0 and r.n_distinct_oos == 10
    assert r.precision == 1.0
    assert set(np.flatnonzero(r.ridge_is).tolist()) == {4, 5, 6} and set(np.flatnonzero(r.ridge_oos).tolist()) == {4, 5}
    assert r.ridge_jaccard == pytest.approx(2 / 3) and r.ridge_shift == pytest.approx(0.5)
    assert (r.pct_best_is, r.pct_best_pooled, r.pct_region, r.pct_pick) == (80.0, 80.0, 90.0, 0.0)
    assert r.null_lift.size == 20 and 1 / 21 <= r.p_lift <= 1.0


def test_null_draws_are_sign_flips_of_the_oos_deviations_scored_with_the_observed_sd():
    """Literal oracle, the 3-cell toy of test_null_signflip (two IS days, four OOS days, blocks of
    two): the four sign patterns rebuild the OOS surface as [4, 4, 7] (observed), [2, 6, 7],
    [8, 4, 3] and [6, 6, 3] - the grid mean stays 5. Every cell's IS net profit is 0, so k = 1 and
    R = [0] (grid order); SD of the observed surface = sqrt(2), so L = (4 - 5) / sqrt(2) and the
    null lifts are (y0 - 5) / sqrt(2) in {-1, -3, 3, 1} / sqrt(2). Pooled OOS [4, 5, 5.5] puts
    cell 2 on top (precision 0); only the two patterns that lift cell 0 put it on top (precision 1)."""
    daily = np.array([[9.0, -9.0, 1, 1, 1, 1], [-9.0, 9.0, 3, 1, -1, 1], [0.0, 0.0, 2, 4, 0, 1]])
    grid, w = _line_grid(daily[:, 0] + daily[:, 1], daily[:, 2:])
    grid.daily_pnl, grid.closed_pnl = daily, daily.copy()                      # keep the toy's own IS days
    r = region_test(grid, [w], n_null=40, seed=3, block=2).windows[0]
    s2 = math.sqrt(2.0)
    assert r.k == 1 and r.region.tolist() == [0] and r.lift == pytest.approx(-1 / s2) and r.precision == 0.0
    lifts = np.round(r.null_lift * s2, 9)
    assert set(lifts.tolist()) == {-1.0, -3.0, 3.0, 1.0}                       # all four patterns drawn, nothing else
    assert r.null_precision.tolist() == [1.0 if v > 0 else 0.0 for v in lifts]   # lift and precision from the same draw
    assert r.p_lift == (1 + int((r.null_lift >= r.lift).sum())) / 41 and r.p_precision == 1.0


def test_pooled_over_the_complete_windows_draw_by_draw():
    oos1 = np.array(_OOS_DISTINCT, dtype=float)
    oos2 = np.array(_OOS_TIED, dtype=float)[:, ::-1] + np.arange(10.0)[:, None] * [0.0, 1.0]   # scored, other numbers
    grid, _ = _line_grid(_X, np.column_stack([oos1, np.array(_X, dtype=float)[:, None] * [1.0, 0.0], oos2, np.array(_OOS_TIED)]))
    days = np.arange(grid.n_days)

    def win(i, is_days, oos_days, complete=True):
        return Window(index=i, label=f"w{i}", is_start=grid.dates[is_days[0]], is_end=grid.dates[is_days[-1]],
                      oos_start=grid.dates[oos_days[0]], oos_end_nominal=grid.dates[oos_days[-1]], oos_end=grid.dates[oos_days[-1]],
                      is_mask=np.isin(days, is_days), oos_mask=np.isin(days, oos_days), complete=complete, grid_row=1, params=(0.0,))

    windows = [win(1, [0, 1], [2, 3]), win(2, [4, 5], [6, 7]), win(3, [6, 7], [8, 9]), win(4, [2, 3], [4, 5], complete=False)]
    r = region_test(grid, windows, n_null=30, seed=0)
    w1, w2, w3, w4 = r.windows
    assert r.n_complete == 3 and not w4.complete and len(r.windows) == 4
    assert math.isnan(w3.precision) and np.isfinite(w1.precision) and np.isfinite(w2.precision)
    assert r.lift == pytest.approx(np.mean([w1.lift, w2.lift, w3.lift]))
    null = np.mean([w1.null_lift, w2.null_lift, w3.null_lift], axis=0)
    assert r.p_lift == (1 + int((null >= r.lift).sum())) / 31
    assert r.precision == pytest.approx(np.mean([w1.precision, w2.precision]))    # the unscored window is left out
    null_p = np.mean([w1.null_precision, w2.null_precision], axis=0)
    assert r.p_precision == (1 + int((null_p >= r.precision).sum())) / 31
    assert r.grid_oos_mean == pytest.approx(np.mean([w1.grid_oos_mean, w2.grid_oos_mean, w3.grid_oos_mean]))
    assert not np.array_equal(w1.null_lift, w2.null_lift)                        # windows draw independently
    nothing = region_test(grid, [windows[3]], n_null=10, seed=0)
    assert nothing.n_complete == 0 and math.isnan(nothing.lift) and math.isnan(nothing.p_lift)


def test_identical_variants_leave_precision_and_jaccard_unscored_but_not_the_lift():
    grid, w = _line_grid(_X, _OOS_TIED, grid_row=0)            # custom windows before the pick is set: no pick
    r = region_test(grid, [w], n_null=20, seed=0).windows[0]
    assert r.n_distinct_oos == 3 and r.k == 2
    assert math.isnan(r.precision) and math.isnan(r.p_precision) and math.isnan(r.ridge_jaccard)
    assert r.lift == pytest.approx(4.0 / math.sqrt(4.2)) and math.isnan(r.pct_pick)


# ---- planted-truth oracles (docs/wfc-region-lift.md, 'Evidence'): 750 days, split at day 500, pass = p < 0.05

def _split(grid, split=500):
    """One complete window: in-sample = the days before split; the pick = the best in-sample net profit."""
    is_mask = np.arange(grid.n_days) < split
    d = grid.dates
    return Window(index=1, label="split", is_start=d[0], is_end=d[split - 1], oos_start=d[split], oos_end_nominal=d[-1],
                  oos_end=d[-1], is_mask=is_mask, oos_mask=~is_mask, complete=True,
                  grid_row=int(np.argmax(grid.daily_pnl[:, is_mask].sum(axis=1))) + 1, params=())


def _pass_rates(structure, shape, trade_p, rhos, seeds, n_null=199):
    """(pointwise pass rate, region-lift pass rate) over the seeds and neighbour-noise correlations."""
    from robustness.null_signflip import SignFlipNull
    from robustness.surface import window_metrics
    from robustness.synthetic_multiwalk import make_planted_grid
    from robustness.wfc_grid import wfc_window
    point = lift = runs = 0
    for rho in rhos:
        for seed in seeds:
            grid = make_planted_grid(shape, structure=structure, rho=rho, trade_p=trade_p, n_days=750, split=500, seed=seed)
            w = _split(grid)
            x, y = window_metrics(grid, w.is_mask).net_profit, window_metrics(grid, w.oos_mask).net_profit
            point += wfc_window(x, y, w, SignFlipNull(grid, "NP"), n_null=n_null, seed=seed).p_value < 0.05
            lift += region_test(grid, [w], n_null=n_null, seed=seed).p_lift < 0.05
            runs += 1
    return point / runs, lift / runs


def test_region_lift_outpowers_the_pointwise_correlation_on_a_narrow_ridge_with_adequate_trades():
    """Oracle, docs/wfc-region-lift.md 'Evidence' (8x8 narrow ridge, neighbour-noise correlation 0.6 and 0.9 as on real
    grids): with ~127 OOS trades per combination the lift passes 0.73-0.87 vs pointwise 0.33-0.50;
    with ~51 trades 0.37-0.47 vs 0.33; with ~20 trades both sit near size (0.13-0.20)."""
    seeds = range(30)
    point, lift = _pass_rates("ridge", (8, 8), 0.5, (0.6, 0.9), seeds)
    assert lift >= 0.6 and lift - point >= 0.2, (point, lift)
    point, lift = _pass_rates("ridge", (8, 8), 0.2, (0.6, 0.9), seeds)
    assert lift > point, (point, lift)
    point, lift = _pass_rates("ridge", (8, 8), 0.08, (0.6, 0.9), seeds)
    assert abs(lift - point) <= 0.15 and lift <= 0.35, (point, lift)


def test_region_lift_size_is_controlled_on_correlated_noise_grids():
    """Oracle for the statistic under its null: nothing planted, neighbours sharing 90% of their
    noise (the generator of test_wfc_grid's size oracle) - the lift must pass at most 7% of the
    time at the 5% level; overlap precision (display only) is conservative (2.6% in the sweep)."""
    from robustness.synthetic_multiwalk import make_planted_grid
    from robustness.windows import custom_windows
    lift = prec = 0
    seeds = range(200)
    for seed in seeds:
        grid = make_planted_grid((6, 6), structure="noise", rho=0.9, trade_p=0.2, n_days=600, seed=seed)
        r = region_test(grid, [custom_windows(grid.dates, "single")[0]], n_null=199, seed=seed)
        lift += r.p_lift < 0.05
        prec += r.p_precision < 0.05
    assert lift <= 0.07 * len(seeds) and prec <= 0.07 * len(seeds), (lift, prec)


def test_a_decayed_surface_never_passes():
    """When the bump flips sign after the split, the in-sample top region is the
    out-of-sample bottom - never significant, and negative on average once trades suffice."""
    from robustness.synthetic_multiwalk import make_planted_grid
    for shape in ((8, 8), (6, 5, 4)):
        for trade_p in (0.2, 0.5):
            lifts = []
            for seed in range(10):
                grid = make_planted_grid(shape, structure="decay", rho=0.9, trade_p=trade_p, n_days=750, split=500, seed=seed)
                r = region_test(grid, [_split(grid)], n_null=199, seed=seed)
                assert r.p_lift >= 0.05, (shape, trade_p, seed, r.lift, r.p_lift)
                lifts.append(r.lift)
            assert np.mean(lifts) < 0, (shape, trade_p, lifts)
