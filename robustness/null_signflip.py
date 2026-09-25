"""Block sign-flip null of a walk-forward window's out-of-sample surface (spec decision 5,
amended 2026-09-25). H0: no parameter combination is expected to differ from the grid average
out of sample. Each OOS day's vector of cell P&L is split into its cross-cell mean (the
strategy's common path, kept as is) and the deviations from it; the deviations of every block
of `block` consecutive OOS trading days get one random sign, and the window metric is
recomputed on the rebuilt daily matrix. A draw therefore keeps every cell's noise covariance,
the edge cells, and identical variants exactly as they are, and needs no grid geometry - the
torus shift of wfc_grid._Shifter breaks all three. Only reason to change: the null's definition."""
from __future__ import annotations

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid
from robustness.surface import metric_from_daily
from robustness.windows import Window

METRICS = ("NP", "NPAvgDD")


class SignFlipNull:
    """Draws of one window's OOS metric surface under H0, with the observed surface's holes kept.

    Accepts the grid (for daily_pnl), the metric name ('NP' or 'NPAvgDD'; anything else raises
    ValueError) and the block length in trading days (>= 1). One instance serves every window
    of one window set; per-window preparation is cached by the window's OOS date range."""

    def __init__(self, grid: MultiWalkGrid, metric: str, block: int = 21):
        if metric not in METRICS:
            raise ValueError(f"unsupported metric {metric!r} (one of {METRICS})")
        if int(block) < 1:
            raise ValueError("block must be at least 1 trading day")
        self.grid, self.metric, self.block = grid, metric, int(block)
        self._cache: dict[tuple, tuple] = {}

    def _prepare(self, window: Window) -> tuple:
        key = (window.index, str(window.oos_start), str(window.oos_end))
        if key not in self._cache:
            d = self.grid.daily_pnl[:, window.oos_mask]
            n_days = d.shape[1]
            common = d.mean(axis=0) if n_days else np.zeros(0)
            dev = d - common[None, :]
            blocks = np.arange(n_days) // self.block
            n_blocks = int(blocks[-1]) + 1 if n_days else 0
            sums = (np.stack([dev[:, blocks == b].sum(axis=1) for b in range(n_blocks)], axis=1)
                    if n_blocks else np.zeros((self.grid.n_iter, 0)))
            self._cache[key] = (common, dev, blocks, n_blocks, sums, float(common.sum()))
        return self._cache[key]

    def draw_one(self, y: np.ndarray, window: Window, rng) -> np.ndarray:
        """One OOS surface under H0.

        Accepts: the observed OOS metric per iteration (NaN = dropped), the Window and a
        Generator (anything with .choice([-1.0, 1.0], size=n_blocks)). Returns: an (n_iter,)
        array, NaN exactly where y is NaN. Guarantees: with every block sign +1 the draw equals
        the observed surface; identical iterations stay identical; the cross-cell mean of every
        OOS day is unchanged."""
        common, dev, blocks, n_blocks, sums, np_common = self._prepare(window)
        signs = np.asarray(rng.choice([-1.0, 1.0], size=n_blocks), dtype=float) if n_blocks else np.zeros(0)
        if self.metric == "NP":
            ynull = np_common + sums @ signs
        else:
            ynull = metric_from_daily(common[None, :] + dev * signs[blocks][None, :], self.metric)
        return np.where(np.isfinite(np.asarray(y, dtype=float)), ynull, np.nan)

    def draws(self, y: np.ndarray, window: Window, rng, n_max: int) -> np.ndarray:
        """n_max independent draws stacked as (n_max, n_iter); deterministic for a given rng."""
        if n_max <= 0:
            return np.empty((0, len(y)))
        return np.stack([self.draw_one(y, window, rng) for _ in range(int(n_max))])
