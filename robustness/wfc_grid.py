"""Walk Forward Correlation on the optimisation grid (Tinsley 2026, SSRN 6324079; spec decision 5).
X = in-sample metric, Y = out-of-sample metric per parameter combination; WFC = rho(X, Y). The gate
follows signal_lab/robustness/wfc.py: Spearman, a null that keeps the surface's smoothness (torus
shifts of the OOS surface on the grid) and the paper's Diagnostic Matrix rule that correlation
alone is not edge (positive OOS among the positive-IS points)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robustness.multiwalk_text import MultiWalkGrid
from robustness.windows import Window


def _finite(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation over the points where both are finite (average ranks for ties).
    Returns 0.0 with fewer than 3 such points or when either side is constant."""
    m = _finite(x, y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return 0.0
    rx = pd.Series(x[m]).rank().to_numpy(); ry = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation over the finite points; 0.0 with fewer than 3 points or a constant side."""
    m = _finite(x, y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])


@dataclass
class WFCWindow:
    index: int
    label: str
    complete: bool
    n_points: int
    n_dropped: int
    x: np.ndarray
    y: np.ndarray
    spearman: float
    pearson: float
    n_pos_is: int
    pos_oos_frac: float
    p_value: float
    null: np.ndarray
    pick_index: int
    pick_is_pct: float
    pick_oos_pct: float
    quadrant: str
    insufficient: bool


@dataclass
class WFCResult:
    metric: str
    windows: list[WFCWindow]
    n_complete: int
    pooled_spearman: float
    pooled_p: float
    pooled_pos_oos_frac: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    n_null: int = 0


class _Shifter:
    """Torus shifts of a surface on a full product grid: shifted[i] = y[cell at grid_pos[i] + s mod shape]."""

    def __init__(self, grid: MultiWalkGrid):
        self.full = grid.is_full_grid
        self.pos = grid.grid_pos
        self.shape = np.asarray(grid.shape)
        if self.full:
            self.table = np.full(tuple(self.shape), -1, dtype=int)
            self.table[tuple(self.pos.T)] = np.arange(len(self.pos))
        self.n_shifts = int(np.prod(self.shape)) - 1

    def all_shifts(self, rng: np.random.Generator, n_max: int) -> list[tuple[int, ...]]:
        if self.n_shifts <= n_max:
            return [s for s in np.ndindex(*self.shape) if any(s)]
        return [self.random_shift(rng) for _ in range(n_max)]

    def random_shift(self, rng: np.random.Generator) -> tuple[int, ...]:
        while True:
            s = tuple(int(rng.integers(n)) for n in self.shape)
            if any(s):
                return s

    def apply(self, y: np.ndarray, shift: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        if not self.full:
            return rng.permutation(y)
        idx = self.table[tuple(((self.pos + np.asarray(shift)) % self.shape).T)]
        return y[idx]


def _quadrant(corr_ok: bool, oos_ok: bool) -> str:
    if corr_ok and oos_ok:
        return "structural edge, low over-fitting"
    if corr_ok:
        return "consistently loss-making strategy"
    if oos_ok:
        return "spurious result, high over-fitting"
    return "noise, no edge"


def _pct_rank(v: np.ndarray, value: float) -> float:
    m = np.isfinite(v)
    return float((v[m] < value).mean() * 100.0) if m.any() and np.isfinite(value) else float("nan")


def wfc_window(x: np.ndarray, y: np.ndarray, window: Window, shifter: _Shifter, *, n_null: int = 999, seed: int = 0,
               min_points: int = 8, min_pos_is: int = 4, alpha: float = 0.05, tau_pos: float = 0.5) -> WFCWindow:
    """WFC for one window. Accepts the per-iteration IS metric x and OOS metric y (NaN = dropped),
    the Window (for the pick and completeness) and a _Shifter over the grid. Returns a WFCWindow
    with the null (torus shifts on a full grid, permutations otherwise) and p = (k+1)/(n+1).
    Guarantees: insufficient when fewer than min_points valid pairs or fewer than min_pos_is
    positive-IS points; then p = 1 and the null is empty."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    valid = _finite(x, y)
    n = int(valid.sum())
    pos_is = valid & (x > 0)
    n_pos_is = int(pos_is.sum())
    pos_oos_frac = float((y[pos_is] > 0).mean()) if n_pos_is else 0.0
    rho, r = spearman(x, y), pearson(x, y)
    insufficient = n < min_points or n_pos_is < min_pos_is
    rng = np.random.default_rng(seed + 1000 * window.index)
    null = np.empty(0)
    if not insufficient:
        yn = np.where(valid, y, np.nan); xn = np.where(valid, x, np.nan)
        shifts = shifter.all_shifts(rng, n_null) if shifter.full else [None] * n_null
        null = np.array([spearman(xn, shifter.apply(yn, s, rng)) for s in shifts])
    p = float((1 + int((null >= rho).sum())) / (1 + null.size)) if null.size else 1.0
    pick = window.grid_row - 1
    return WFCWindow(index=window.index, label=window.label, complete=window.complete, n_points=n, n_dropped=int(len(x) - n),
                     x=x, y=y, spearman=rho, pearson=r, n_pos_is=n_pos_is, pos_oos_frac=pos_oos_frac, p_value=p, null=null,
                     pick_index=pick, pick_is_pct=_pct_rank(x, x[pick]), pick_oos_pct=_pct_rank(y, y[pick]),
                     quadrant=_quadrant(p < alpha, pos_oos_frac >= tau_pos), insufficient=insufficient)


def wfc_test(pairs: list[tuple[np.ndarray, np.ndarray]], windows: list[Window], grid: MultiWalkGrid, *, metric: str,
             alpha: float = 0.05, tau_pos: float = 0.5, n_null: int = 999, seed: int = 0) -> WFCResult:
    """WFC over all windows plus the pooled gate.

    Accepts: pairs[i] = (IS metric, OOS metric) arrays for windows[i]; the grid (for shifts).
    Returns: WFCResult with one WFCWindow per window; pooled statistic = mean Spearman over the
    complete, sufficient windows; pooled null = n_null draws of one independent shift per such
    window; passed iff pooled p < alpha and pooled positive-OOS share >= tau_pos.
    Guarantees: with no usable window, passed is False, reasons == ['wfc_insufficient'] and the
    pooled values are NaN; deterministic for a given seed."""
    shifter = _Shifter(grid)
    results = [wfc_window(x, y, w, shifter, n_null=n_null, seed=seed, alpha=alpha, tau_pos=tau_pos)
               for (x, y), w in zip(pairs, windows)]
    usable = [(r, p) for r, p in zip(results, pairs) if r.complete and not r.insufficient]
    if not usable:
        return WFCResult(metric=metric, windows=results, n_complete=sum(r.complete for r in results), pooled_spearman=float("nan"),
                         pooled_p=float("nan"), pooled_pos_oos_frac=float("nan"), passed=False, reasons=["wfc_insufficient"], n_null=n_null)
    pooled_rho = float(np.mean([r.spearman for r, _ in usable]))
    pooled_pos = float(np.mean([r.pos_oos_frac for r, _ in usable]))
    rng = np.random.default_rng(seed + 99)
    draws = np.empty(n_null)
    for b in range(n_null):
        vals = []
        for r, (x, y) in usable:
            yn = np.where(np.isfinite(y), y, np.nan)
            vals.append(spearman(x, shifter.apply(yn, shifter.random_shift(rng), rng)))
        draws[b] = np.mean(vals)
    pooled_p = float((1 + int((draws >= pooled_rho).sum())) / (1 + n_null))
    reasons = []
    if pooled_p >= alpha:
        reasons.append("wfc_low_corr")
    if pooled_pos < tau_pos:
        reasons.append("wfc_no_oos_edge")
    return WFCResult(metric=metric, windows=results, n_complete=sum(r.complete for r in results), pooled_spearman=pooled_rho,
                     pooled_p=pooled_p, pooled_pos_oos_frac=pooled_pos, passed=not reasons, reasons=reasons, n_null=n_null)
