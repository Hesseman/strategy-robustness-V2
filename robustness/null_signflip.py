"""The nulls of a walk-forward window's out-of-sample surface (spec decision 5, amended 2026-09-25;
why and evidence: docs/wfc-region-lift.md, 'The null'). H0: no parameter combination is expected
to differ from the grid average out of sample. Each OOS day's vector of cell P&L is split into its
cross-cell mean (the strategy's common path, kept as is) and the deviations from it, cut into
blocks of `block` consecutive OOS trading days.

- SignFlipNull, the gate's null: every block's deviations get one random sign, and the window
  metric is recomputed on the rebuilt daily matrix. It serves both the region lift (region_wfc,
  the WFC gate) and Tinsley's correlation (wfc_grid, the continuity number). A draw keeps every
  cell's noise covariance, the edge cells, and identical variants exactly as they are, and needs
  no grid geometry - the torus shift of wfc_grid._Shifter breaks all three. Exact when each
  block's deviation vector is as likely as its negation.
- BlockBootstrapNull, its cross-check: the same blocks, resampled with replacement instead of
  flipped, after each cell's deviations are centred on zero. It keeps the blocks' skew, which the
  flip symmetrises away. Net profit only.

Only reason to change: the nulls' definitions."""
from __future__ import annotations

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid
from robustness.surface import metric_from_daily
from robustness.windows import Window

METRICS = ("NP", "NPAvgDD")


class _BlockNull:
    """What the two nulls share: the grid, the metric (one of the class's METRICS; anything else
    raises ValueError), the block length in trading days (>= 1) and the per-window split of the OOS
    days into common path, deviations and block sums. One instance serves every window of one
    window set; per-window preparation is cached by the window's OOS date range."""

    METRICS: tuple[str, ...] = ()

    def __init__(self, grid: MultiWalkGrid, metric: str, block: int = 21):
        if metric not in self.METRICS:
            raise ValueError(f"unsupported metric {metric!r} (one of {self.METRICS})")
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
        raise NotImplementedError

    def draws(self, y: np.ndarray, window: Window, rng, n_max: int) -> np.ndarray:
        """n_max independent draws stacked as (n_max, n_iter); deterministic for a given rng."""
        if n_max <= 0:
            return np.empty((0, len(y)))
        return np.stack([self.draw_one(y, window, rng) for _ in range(int(n_max))])


class SignFlipNull(_BlockNull):
    """Draws of one window's OOS metric surface under H0, with the observed surface's holes kept.

    Accepts the grid (for daily_pnl), the metric name ('NP' or 'NPAvgDD'; anything else raises
    ValueError) and the block length in trading days (>= 1)."""

    METRICS = METRICS

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


class BlockBootstrapNull(_BlockNull):
    """The sign-flip's cross-check: draws of one window's OOS net profit surface under H0 by a
    centred block bootstrap of the same deviations.

    Accepts the grid, the metric ('NP' only - a resampled block can be shorter than the position
    it fills, the window's last block, so a path metric has no rebuilt path; anything else raises
    ValueError) and the block length in trading days (>= 1). Each combination's deviations are
    centred per day: its mean daily deviation, (its OOS net profit - the grid's) / OOS days, is
    taken off every day. A draw fills each of the window's block positions with a block picked
    uniformly with replacement from those centred blocks, on top of the common path."""

    METRICS = ("NP",)

    def __init__(self, grid: MultiWalkGrid, metric: str = "NP", block: int = 21):
        super().__init__(grid, metric, block)
        self._centred_cache: dict[tuple, np.ndarray] = {}

    def _centred(self, window: Window) -> np.ndarray:
        """(n_iter, n_blocks) block sums of the per-day centred deviations."""
        key = (window.index, str(window.oos_start), str(window.oos_end))
        if key not in self._centred_cache:
            _, dev, blocks, n_blocks, sums, _ = self._prepare(window)
            n_days = dev.shape[1]
            mean_day = dev.sum(axis=1) / n_days if n_days else np.zeros(dev.shape[0])
            self._centred_cache[key] = sums - mean_day[:, None] * np.bincount(blocks, minlength=n_blocks)[None, :]
        return self._centred_cache[key]

    def draw_one(self, y: np.ndarray, window: Window, rng) -> np.ndarray:
        """One OOS net profit surface under H0.

        Accepts: the observed OOS net profit per iteration (NaN = dropped), the Window and a
        Generator (anything with .integers(0, n_blocks, size=n_blocks)). Returns: an (n_iter,)
        array, NaN exactly where y is NaN. Guarantees: averaged over the equally likely picks, every
        iteration's draw is the common path's net profit (H0); identical iterations stay
        identical; the cross-cell mean of every draw is the common path's net profit."""
        np_common = self._prepare(window)[5]
        centred = self._centred(window)
        n_blocks = centred.shape[1]
        picks = np.asarray(rng.integers(0, n_blocks, size=n_blocks), dtype=int) if n_blocks else np.zeros(0, dtype=int)
        ynull = np_common + centred @ np.bincount(picks, minlength=n_blocks)
        return np.where(np.isfinite(np.asarray(y, dtype=float)), ynull, np.nan)
