"""Walk Forward Correlation on the optimisation grid (Tinsley 2026, SSRN 6324079; spec decision 5).
X = in-sample metric, Y = out-of-sample metric per parameter combination; WFC = rho(X, Y). Since
2026-09-25 this is the WFC card's continuity number, not its gate: the battery's gate is the region
lift (region_wfc; docs/wfc-region-lift.md). WFCResult.passed keeps the paper's rule for comparison -
Spearman, a null, and the Diagnostic Matrix rule that correlation alone is not edge (positive OOS
among the positive-IS points), as ported from signal_lab/robustness/wfc.py. The null (amended
2026-09-25) is the block sign-flip of the OOS daily cross-sectional deviations
(null_signflip.SignFlipNull): it keeps every cell's noise covariance, edges and ties. The torus
shift of the OOS surface on the grid is kept behind null='torus' for comparison only - it passes
10-30% of no-structure grids once neighbours share their noise, which real grids always do."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robustness.multiwalk_text import MultiWalkGrid
from robustness.null_signflip import SignFlipNull
from robustness.windows import Window

NULLS = ("signflip", "torus")


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
    null: str = "signflip"


class _Shifter:
    """Torus shifts of a surface on a full product grid: shifted[i] = y[cell at grid_pos[i] + s mod shape].
    Kept as the null='torus' generator; permutations when the grid is not a full product grid."""

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

    def draw_one(self, y: np.ndarray, window: Window, rng: np.random.Generator) -> np.ndarray:
        """One shifted (or, off a full grid, permuted) copy of y; the window is not needed."""
        return self.apply(y, self.random_shift(rng), rng)

    def draws(self, y: np.ndarray, window: Window, rng: np.random.Generator, n_max: int) -> np.ndarray:
        """Every non-zero shift when there are at most n_max of them, else n_max random ones,
        stacked as (k, n_iter); permutations off a full grid."""
        shifts = self.all_shifts(rng, n_max) if self.full else [None] * n_max
        return np.stack([self.apply(y, s, rng) for s in shifts]) if shifts else np.empty((0, len(y)))


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


def wfc_window(x: np.ndarray, y: np.ndarray, window: Window, null_gen, *, n_null: int = 999, seed: int = 0,
               min_points: int = 8, min_pos_is: int = 4, alpha: float = 0.05, tau_pos: float = 0.5) -> WFCWindow:
    """WFC for one window. Accepts the per-iteration IS metric x and OOS metric y (NaN = dropped),
    the Window (for the pick and completeness) and a null generator (SignFlipNull or _Shifter:
    anything with draws(y, window, rng, n_max) -> (k, n_iter)). Returns a WFCWindow with the null
    draws' Spearman values and p = (k+1)/(n+1); the rng is seeded seed + 1000 * window.index.
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
        yn = np.where(np.isfinite(y), y, np.nan)   # Y's surface with its own holes; correlate over the cells finite on both sides of each draw (identical construction to the pooled null)
        null = np.array([spearman(x, d) for d in null_gen.draws(yn, window, rng, n_null)])
    p = float((1 + int((null >= rho).sum())) / (1 + null.size)) if null.size else 1.0
    pick = window.grid_row - 1
    return WFCWindow(index=window.index, label=window.label, complete=window.complete, n_points=n, n_dropped=int(len(x) - n),
                     x=x, y=y, spearman=rho, pearson=r, n_pos_is=n_pos_is, pos_oos_frac=pos_oos_frac, p_value=p, null=null,
                     pick_index=pick, pick_is_pct=_pct_rank(x, x[pick]), pick_oos_pct=_pct_rank(y, y[pick]),
                     quadrant=_quadrant(p < alpha, pos_oos_frac >= tau_pos), insufficient=insufficient)


def top_n_default(n_valid: int) -> int:
    """How many of the best in-sample combinations to follow out-of-sample: 10, but never more
    than a third of the valid points (at least 1)."""
    return min(10, max(1, n_valid // 3))


@dataclass
class TopN:
    n: int
    indices: list[int]            # iteration indices, best in-sample first
    median_oos_rank: float        # 1 = best out-of-sample among the valid points
    n_valid: int
    n_positive_oos: int


def top_n_summary(w: WFCWindow, n: int | None = None) -> TopN:
    """Where the best in-sample combinations landed out-of-sample.

    Accepts: a WFCWindow and n (default top_n_default of the valid points).
    Returns: TopN over the points finite on both sides - the n best by in-sample metric (ties
    keep grid order), their median out-of-sample rank (1 = best, average ranks for ties) and how
    many were profitable out-of-sample.
    Guarantees: n <= valid points; n = 0 with a NaN median when nothing is valid. The WFC
    correlation itself still uses every combination - this only follows the winners."""
    idx = np.flatnonzero(_finite(w.x, w.y))
    if idx.size == 0:
        return TopN(0, [], float("nan"), 0, 0)
    k = min(top_n_default(idx.size) if n is None else n, idx.size)
    best = idx[np.argsort(-w.x[idx], kind="stable")][:k]
    oos_rank = dict(zip(idx.tolist(), pd.Series(-w.y[idx]).rank().to_numpy()))
    return TopN(n=int(k), indices=best.tolist(), median_oos_rank=float(np.median([oos_rank[i] for i in best])),
                n_valid=int(idx.size), n_positive_oos=int((w.y[best] > 0).sum()))


@dataclass
class Band:
    band: int                     # 1 = the best tenth in-sample
    n: int
    is_mean: float
    oos_mean: float
    oos_se: float                 # standard error of oos_mean; NaN for a single point
    oos_pos_share: float


def wfc_bands(w: WFCWindow, n_bands: int = 10) -> list[Band]:
    """The in-sample ranking cut into n_bands bands (band 1 = best) with each band's mean
    out-of-sample metric - readable at any grid size, where a scatter of thousands of points is not.

    Accepts: a WFCWindow and the number of bands. Returns: one Band per band over the points
    finite on both sides; [] when there are fewer valid points than bands.
    Guarantees: every valid point is in exactly one band and band sizes differ by at most one."""
    idx = np.flatnonzero(_finite(w.x, w.y))
    if idx.size < n_bands:
        return []
    order = idx[np.argsort(-w.x[idx], kind="stable")]
    out = []
    for b, chunk in enumerate(np.array_split(order, n_bands), 1):
        y = w.y[chunk]
        se = float(y.std(ddof=1) / np.sqrt(y.size)) if y.size > 1 else float("nan")
        out.append(Band(band=b, n=int(chunk.size), is_mean=float(w.x[chunk].mean()), oos_mean=float(y.mean()),
                        oos_se=se, oos_pos_share=float((y > 0).mean())))
    return out


def wfc_test(pairs: list[tuple[np.ndarray, np.ndarray]], windows: list[Window], grid: MultiWalkGrid, *, metric: str,
             alpha: float = 0.05, tau_pos: float = 0.5, n_null: int = 999, seed: int = 0, null: str = "signflip",
             block: int = 21) -> WFCResult:
    """WFC over all windows plus the pooled gate.

    Accepts: pairs[i] = (IS metric, OOS metric) arrays for windows[i]; the grid (daily P&L for the
    sign-flip null, geometry for the torus null); null in NULLS ('signflip' default, 'torus' for
    comparison; anything else raises ValueError); block = sign-flip block length in trading days.
    Returns: WFCResult with one WFCWindow per window; pooled statistic = mean Spearman over the
    complete, sufficient windows; pooled null = n_null draws of one independent null surface per
    such window; passed iff pooled p < alpha and pooled positive-OOS share >= tau_pos.
    Guarantees: with no usable window, passed is False, reasons == ['wfc_insufficient'] and the
    pooled values are NaN; deterministic for a given seed."""
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    null_gen = SignFlipNull(grid, metric, block) if null == "signflip" else _Shifter(grid)
    results = [wfc_window(x, y, w, null_gen, n_null=n_null, seed=seed, alpha=alpha, tau_pos=tau_pos)
               for (x, y), w in zip(pairs, windows)]
    usable = [(r, p, w) for r, p, w in zip(results, pairs, windows) if r.complete and not r.insufficient]
    if not usable:
        return WFCResult(metric=metric, windows=results, n_complete=sum(r.complete for r in results), pooled_spearman=float("nan"),
                         pooled_p=float("nan"), pooled_pos_oos_frac=float("nan"), passed=False, reasons=["wfc_insufficient"],
                         n_null=n_null, null=null)
    pooled_rho = float(np.mean([r.spearman for r, _, _ in usable]))
    pooled_pos = float(np.mean([r.pos_oos_frac for r, _, _ in usable]))
    rng = np.random.default_rng(seed + 99)
    draws = np.empty(n_null)
    for b in range(n_null):
        vals = []
        for r, (x, y), w in usable:
            yn = np.where(np.isfinite(y), y, np.nan)
            vals.append(spearman(x, null_gen.draw_one(yn, w, rng)))
        draws[b] = np.mean(vals)
    pooled_p = float((1 + int((draws >= pooled_rho).sum())) / (1 + n_null))
    reasons = []
    if pooled_p >= alpha:
        reasons.append("wfc_low_corr")
    if pooled_pos < tau_pos:
        reasons.append("wfc_no_oos_edge")
    return WFCResult(metric=metric, windows=results, n_complete=sum(r.complete for r in results), pooled_spearman=pooled_rho,
                     pooled_p=pooled_p, pooled_pos_oos_frac=pooled_pos, passed=not reasons, reasons=reasons, n_null=n_null, null=null)
