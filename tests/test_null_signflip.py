"""Block sign-flip null of a window's out-of-sample surface (spec decision 5, amended 2026-09-25).
Each OOS day's vector of cell P&L is split into the grid mean (kept) and the deviations (sign
flipped per block), so a draw keeps the strategy's common equity path and every cell's noise
covariance, edge and tie structure, and tests 'no cell is expected to beat the grid average'."""
import numpy as np
import pandas as pd
import pytest

from robustness.multiwalk_text import MultiWalkGrid
from robustness.null_signflip import SignFlipNull
from robustness.surface import daily_drawdown_stats, metric_values, window_metrics
from robustness.windows import Window


def _grid(daily: np.ndarray) -> MultiWalkGrid:
    daily = np.asarray(daily, dtype=float)
    n, t = daily.shape
    dates = pd.bdate_range("2021-01-04", periods=t)
    pos = np.arange(n)[:, None]
    ed = [dates.values[daily[i] != 0].astype("datetime64[D]") for i in range(n)]
    return MultiWalkGrid(param_names=["A"], params=pos.astype(float), axes=[np.arange(n, dtype=float)], grid_pos=pos,
                         dates=dates, daily_pnl=daily, closed_pnl=daily.copy(), exit_dates=ed,
                         exit_pnl=[daily[i][daily[i] != 0] for i in range(n)])


def _window(grid: MultiWalkGrid, split: int) -> Window:
    is_mask = np.arange(grid.n_days) < split
    return Window(index=1, label="w", is_start=grid.dates[0], is_end=grid.dates[split - 1], oos_start=grid.dates[split],
                  oos_end_nominal=grid.dates[-1], oos_end=grid.dates[-1], is_mask=is_mask, oos_mask=~is_mask, complete=True,
                  grid_row=1, params=(0.0,))


class _Signs:
    """Stands in for the generator: hands out a fixed sign per block."""

    def __init__(self, signs):
        self.signs = np.asarray(signs, dtype=float)

    def choice(self, a, size=None, **kwargs):
        return self.signs[:size].copy()


# Six trading days, the last four out of sample, blocks of two days.
#   OOS daily P&L      c0 [1, 1, 1, 1]   c1 [3, 1, -1, 1]   c2 [2, 4, 0, 1]
#   grid mean per day  [2, 2, 0, 1]  -> common net profit 5
#   deviations         c0 [-1, -1, 1, 0]  c1 [1, -1, -1, 0]  c2 [0, 2, 0, 0]
#   block sums (b1, b2) c0 (-2, 1)  c1 (0, -1)  c2 (2, 0)
_DAILY = np.array([[9.0, -9.0, 1, 1, 1, 1], [-9.0, 9.0, 3, 1, -1, 1], [0.0, 0.0, 2, 4, 0, 1]])
_PATTERNS = {(1, 1): [4.0, 4.0, 7.0], (1, -1): [2.0, 6.0, 7.0], (-1, 1): [8.0, 4.0, 3.0], (-1, -1): [6.0, 6.0, 3.0]}


def test_np_draw_is_the_common_path_plus_signed_block_deviations():
    grid = _grid(_DAILY)
    w = _window(grid, 2)
    y = metric_values(window_metrics(grid, w.oos_mask), "NP")
    assert y.tolist() == _PATTERNS[(1, 1)]
    null = SignFlipNull(grid, "NP", block=2)
    for signs, expected in _PATTERNS.items():
        assert null.draw_one(y, w, _Signs(signs)).tolist() == expected
    draws = null.draws(y, w, np.random.default_rng(0), 40)
    assert draws.shape == (40, 3)
    seen = {tuple(row) for row in draws.tolist()}
    assert seen <= {tuple(v) for v in _PATTERNS.values()} and len(seen) == 4


def test_identical_cells_are_never_separated():
    grid = _grid(np.tile([[2.0, -1.0, 3.0, -2.0, 1.0, 4.0, -3.0, 2.0]], (4, 1)))
    w = _window(grid, 3)
    y = metric_values(window_metrics(grid, w.oos_mask), "NP")
    draws = SignFlipNull(grid, "NP", block=2).draws(y, w, np.random.default_rng(1), 25)
    assert np.array_equal(draws, np.tile(y, (25, 1)))


def test_npavgdd_draw_matches_a_scalar_recomputation_of_the_flipped_path():
    rng = np.random.default_rng(7)
    grid = _grid(np.round(rng.normal(size=(5, 60)) * 100, 2))
    w = _window(grid, 30)
    y = metric_values(window_metrics(grid, w.oos_mask), "NPAvgDD")
    null = SignFlipNull(grid, "NPAvgDD", block=7)
    assert np.allclose(null.draw_one(y, w, _Signs([1] * 5)), y)
    signs = np.array([1, -1, -1, 1, -1], dtype=float)              # 30 OOS days / block 7 -> 5 blocks, last one 2 days
    d = grid.daily_pnl[:, w.oos_mask]
    common = d.mean(axis=0)
    per_day = np.repeat(signs, 7)[:30]
    expected = []
    for i in range(5):
        net, _, avg_dd = daily_drawdown_stats(common + per_day * (d[i] - common))
        expected.append(net / -avg_dd)
    assert np.allclose(null.draw_one(y, w, _Signs(signs)), expected)


def test_holes_in_the_observed_surface_stay_holes():
    grid = _grid(np.random.default_rng(3).normal(size=(6, 40)))
    w = _window(grid, 20)
    y = metric_values(window_metrics(grid, w.oos_mask), "NP")
    y[[1, 4]] = np.nan
    draws = SignFlipNull(grid, "NP").draws(y, w, np.random.default_rng(0), 10)
    assert np.isnan(draws[:, [1, 4]]).all() and np.isfinite(np.delete(draws, [1, 4], axis=1)).all()


def test_draws_follow_the_generator_and_the_block_length():
    grid = _grid(np.random.default_rng(5).normal(size=(3, 70)))
    w = _window(grid, 20)                                            # 50 OOS days
    y = metric_values(window_metrics(grid, w.oos_mask), "NP")
    null = SignFlipNull(grid, "NP", block=21)                       # -> 3 blocks, 2^3 sign patterns
    a = null.draws(y, w, np.random.default_rng(11), 200)
    b = null.draws(y, w, np.random.default_rng(11), 200)
    assert np.array_equal(a, b) and not np.array_equal(a, null.draws(y, w, np.random.default_rng(12), 200))
    assert 5 <= len({tuple(np.round(r, 6)) for r in a.tolist()}) <= 8
    assert null.block == 21


def test_metric_must_be_one_the_surface_knows():
    grid = _grid(np.ones((2, 6)))
    with pytest.raises(ValueError):
        SignFlipNull(grid, "Sharpe")
