"""Region lift on the optimisation grid (docs/research/2026-09-25-wfc-region-concordance.md § 6; spec
decision 5, amended 2026-09-25). Tinsley's WFC correlates every cell, so on a mostly flat grid the
flat majority dilutes a ridge. The region layer asks the trader's question instead: does the
in-sample top region beat the grid average out of sample? Per window, on net profit: the pooled
in-sample surface (each cell = the equal-weight mean of itself and its Chebyshev-1 neighbours, no
wrap) picks the region R = its top Q_TOP of cells; the lift L = (mean OOS net profit over R - the
grid mean) / the cross-cell SD of OOS net profit, scored on the raw OOS surface against the block
sign-flip null (null_signflip.SignFlipNull). For display: the overlap precision of R with the pooled
OOS top Q_TOP, the largest connected top-Q_TOP component in vs out of sample (Jaccard, centroid
shift), and where four pick rules landed out of sample. Deliberately absent: a correlation of the
pooled surfaces (§ 4b: weaker than the pointwise one on real-like noise). No per-cell min-trade
hole: the null prices each cell's noise. Only reason to change: the region method's definition."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid
from robustness.null_signflip import SignFlipNull
from robustness.plateau_grid import neighbours
from robustness.surface import window_metrics
from robustness.windows import Window

Q_TOP = 0.2   # share of the grid in the region; fixed a priori (§ 6.8), never tuned per project


def neighbourhoods(grid_pos: np.ndarray) -> list[np.ndarray]:
    """Accepts the (n, k) grid positions. Returns, per combination, its own index followed by every
    combination at Chebyshev distance 1 (plateau_grid.neighbours); the grid does not wrap.
    Guarantees one int array per combination, its own index first."""
    pos = np.asarray(grid_pos)
    return [np.concatenate([[i], neighbours(pos, i)]).astype(int) for i in range(len(pos))]


def pool(values: np.ndarray, nbhd: list[np.ndarray]) -> np.ndarray:
    """Equal-weight mean of each neighbourhood.

    Accepts: values (n,) or (draws, n) and neighbourhoods() of the same n combinations.
    Returns: an array of the same shape, pooled along the last axis.
    Guarantees: linear (pool(a + b) = pool(a) + pool(b)) and a constant stays that constant."""
    v = np.asarray(values, dtype=float)
    width = max(len(j) for j in nbhd)
    idx = np.array([np.pad(j, (0, width - len(j)), constant_values=j[0]) for j in nbhd])
    real = np.array([np.arange(width) < len(j) for j in nbhd])
    count = real.sum(axis=1)
    if v.ndim == 1:
        return np.where(real, v[idx], 0.0).sum(axis=1) / count
    out = np.empty_like(v)
    step = max(1, 4_000_000 // max(1, idx.size))          # bound the (draws, n, width) gather to ~32 MB
    for s in range(0, v.shape[0], step):
        out[s:s + step] = np.where(real, v[s:s + step][:, idx], 0.0).sum(axis=2) / count
    return out


def top_set(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, best first; ties keep grid order (NaN ranks last)."""
    v = np.asarray(values, dtype=float)
    return np.argsort(-np.where(np.isfinite(v), v, -np.inf), kind="stable")[:k]


def largest_component(mask: np.ndarray, grid_pos: np.ndarray) -> np.ndarray:
    """Accepts a boolean mask over the combinations and their (n, k) grid positions. Returns the
    boolean mask of the largest set of masked combinations connected through Chebyshev-1 steps
    (diagonals connect); on a tie, the component holding the lowest index.
    Guarantees a subset of mask; all False when mask is empty."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    out = np.zeros(mask.size, dtype=bool)
    if idx.size == 0:
        return out
    pos = [tuple(p) for p in np.asarray(grid_pos)[idx].tolist()]
    at: dict[tuple, list[int]] = {}
    for j, p in enumerate(pos):
        at.setdefault(p, []).append(j)
    steps = list(itertools.product((-1, 0, 1), repeat=len(pos[0])))
    comp = np.full(idx.size, -1)
    for start in range(idx.size):                                # ascending: a component is labelled by its lowest member
        if comp[start] >= 0:
            continue
        comp[start] = start
        stack = [start]
        while stack:
            u = stack.pop()
            for s in steps:
                for v in at.get(tuple(a + b for a, b in zip(pos[u], s)), ()):
                    if comp[v] < 0:
                        comp[v] = start
                        stack.append(v)
    labels, sizes = np.unique(comp, return_counts=True)
    out[idx[comp == labels[int(np.argmax(sizes))]]] = True
    return out


def ridge(values: np.ndarray, grid_pos: np.ndarray, q: float = Q_TOP) -> np.ndarray:
    """The ridge of a surface: the largest connected component (largest_component) of the
    combinations at or above the surface's (1 - q) quantile (linear interpolation, NaN ignored).
    Returns a boolean mask; all False when no value is finite."""
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).any():
        return np.zeros(v.size, dtype=bool)
    return largest_component(np.isfinite(v) & (v >= np.nanquantile(v, 1 - q)), grid_pos)


def region_size(n: int) -> int:
    """|R| for a grid of n combinations: round(Q_TOP x n), at least 1."""
    return max(1, int(round(Q_TOP * n)))


def distinct_patterns(daily: np.ndarray) -> int:
    """Number of distinct rows of an (n, days) daily $ P&L matrix rounded to cents - identical
    variants count once. A matrix without days has one pattern (every row is the same)."""
    d = np.round(np.asarray(daily, dtype=float), 2) + 0.0          # + 0.0 folds -0.0 into 0.0
    if d.shape[0] == 0:
        return 0
    return 1 if d.shape[1] == 0 else int(np.unique(d, axis=0).shape[0])


def _pct_rank(v: np.ndarray, value: float) -> float:
    """Share of combinations with a lower value, in % (100 = better than every one); NaN for a non-finite value."""
    return float((v < value).mean() * 100.0) if np.isfinite(value) else float("nan")


def _p(null: np.ndarray, observed: float) -> float:
    """(k + 1) / (n + 1) with k = null draws at least as large; NaN without draws or observation."""
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + int((null >= observed).sum())) / (1 + null.size))


def _mean(values: list[float]) -> float:
    """Mean of the finite values; NaN when there are none."""
    f = [v for v in values if np.isfinite(v)]
    return float(np.mean(f)) if f else float("nan")


@dataclass
class RegionWindow:
    index: int
    label: str
    complete: bool
    k: int                       # region size |R|
    x: np.ndarray                # in-sample net profit per combination
    y: np.ndarray                # out-of-sample net profit per combination
    x_pooled: np.ndarray         # neighbourhood means of x
    y_pooled: np.ndarray         # neighbourhood means of y
    region: np.ndarray           # R: the k best of x_pooled, best first
    lift: float                  # L: (region_oos_mean - grid_oos_mean) / oos_sd
    region_oos_mean: float
    grid_oos_mean: float
    oos_sd: float                # cross-cell SD of y (ddof 0)
    oos_positive_share: float    # share of combinations with y > 0
    null_lift: np.ndarray
    p_lift: float
    n_distinct_oos: int          # distinct out-of-sample daily patterns
    precision: float             # share of R in the top k of y_pooled; NaN when n_distinct_oos < 2k
    null_precision: np.ndarray
    p_precision: float
    ridge_is: np.ndarray         # ridge() of x_pooled
    ridge_oos: np.ndarray        # ridge() of y_pooled
    ridge_jaccard: float         # NaN when n_distinct_oos < 2k
    ridge_shift: float           # centroid distance of the two ridges, grid steps
    pct_best_is: float           # OOS percentile of the best in-sample combination
    pct_best_pooled: float       # ... of the best pooled in-sample combination (R's first)
    pct_region: float            # ... of the region ensemble (the mean of y over R)
    pct_pick: float              # ... of the window's pick; NaN without one


@dataclass
class RegionResult:
    windows: list[RegionWindow]
    n_complete: int
    k: int
    q: float
    block: int
    n_null: int
    lift: float                  # pooled over the complete windows: mean L
    p_lift: float
    region_oos_mean: float
    grid_oos_mean: float
    oos_positive_share: float
    precision: float             # mean over the complete windows where it is scored
    p_precision: float
    ridge_jaccard: float
    ridge_shift: float
    pct_best_is: float
    pct_best_pooled: float
    pct_region: float
    pct_pick: float


def _region_window(grid: MultiWalkGrid, w: Window, nbhd: list[np.ndarray], null: SignFlipNull, n_null: int,
                   seed: int) -> RegionWindow:
    n = grid.n_iter
    k = region_size(n)
    x = window_metrics(grid, w.is_mask).net_profit.astype(float)
    y = window_metrics(grid, w.oos_mask).net_profit.astype(float)
    xp, yp = pool(x, nbhd), pool(y, nbhd)
    region = top_set(xp, k)
    sd = float(np.std(y))

    def lift_of(v: np.ndarray) -> np.ndarray:
        return (v[..., region].mean(axis=-1) - v.mean(axis=-1)) / sd if sd > 0 else np.zeros(v.shape[:-1])

    in_region = np.zeros(n, dtype=bool)
    in_region[region] = True

    def precision_of(v: np.ndarray) -> float:
        return float(in_region[top_set(v, k)].mean())

    draws = null.draws(y, w, np.random.default_rng(seed + 1000 * w.index), n_null)
    lift, null_lift = float(lift_of(y)), np.asarray(lift_of(draws), dtype=float).reshape(-1)
    n_distinct = distinct_patterns(grid.daily_pnl[:, w.oos_mask])
    scored = n_distinct >= 2 * k
    precision = precision_of(yp) if scored else float("nan")
    null_precision = np.array([precision_of(d) for d in pool(draws, nbhd)]) if len(draws) else np.empty(0)
    ridge_is, ridge_oos = ridge(xp, grid.grid_pos), ridge(yp, grid.grid_pos)
    union = int((ridge_is | ridge_oos).sum())
    pos = np.asarray(grid.grid_pos, dtype=float)
    return RegionWindow(
        index=w.index, label=w.label, complete=w.complete, k=k, x=x, y=y, x_pooled=xp, y_pooled=yp, region=region,
        lift=lift, region_oos_mean=float(y[region].mean()), grid_oos_mean=float(y.mean()), oos_sd=sd,
        oos_positive_share=float((y > 0).mean()), null_lift=null_lift, p_lift=_p(null_lift, lift),
        n_distinct_oos=n_distinct, precision=precision, null_precision=null_precision,
        p_precision=_p(null_precision, precision), ridge_is=ridge_is, ridge_oos=ridge_oos,
        ridge_jaccard=float((ridge_is & ridge_oos).sum() / union) if union and scored else float("nan"),
        ridge_shift=(float(np.linalg.norm(pos[ridge_is].mean(axis=0) - pos[ridge_oos].mean(axis=0)))
                     if ridge_is.any() and ridge_oos.any() else float("nan")),
        pct_best_is=_pct_rank(y, y[int(np.argmax(x))]), pct_best_pooled=_pct_rank(y, y[region[0]]),
        pct_region=_pct_rank(y, float(y[region].mean())),
        pct_pick=_pct_rank(y, y[w.grid_row - 1]) if 1 <= w.grid_row <= n else float("nan"))


def region_test(grid: MultiWalkGrid, windows: list[Window], *, n_null: int = 999, seed: int = 0,
                block: int = 21) -> RegionResult:
    """The region layer over every window, pooled over the complete ones.

    Accepts: the grid, its windows (a window's pick is grid_row, 1-based; 0 = none), n_null
    sign-flip draws per window, seed, and the sign-flip block length in trading days.
    Returns: RegionResult with one RegionWindow per window, in order. A window's null is
    SignFlipNull(grid, 'NP', block).draws of the raw OOS net profit, seeded seed + 1000 x window
    index; a draw's lift is scaled by the observed SD (a fixed unit per window, as in the note's
    prototype) and its precision is taken on the pooled draw. Pooled over the complete windows:
    the mean of each statistic and, for L and precision, the per-window draws averaged draw by
    draw (the windows' draws are independent); p = (k + 1) / (n + 1). Precision pools only the
    windows where it is scored; display values are means of the finite ones.
    Guarantees: deterministic for a seed; pooled values NaN when no window is complete; nothing
    depends on a per-cell trade count."""
    nbhd = neighbourhoods(grid.grid_pos)
    null = SignFlipNull(grid, "NP", block)
    rws = [_region_window(grid, w, nbhd, null, int(n_null), seed) for w in windows]
    use = [r for r in rws if r.complete]

    def pooled(stat: str, null_name: str) -> tuple[float, float]:
        ok = [r for r in use if np.isfinite(getattr(r, stat))]
        if not ok:
            return float("nan"), float("nan")
        obs = float(np.mean([getattr(r, stat) for r in ok]))
        return obs, _p(np.mean(np.stack([getattr(r, null_name) for r in ok]), axis=0), obs)

    lift, p_lift = pooled("lift", "null_lift")
    precision, p_precision = pooled("precision", "null_precision")
    return RegionResult(
        windows=rws, n_complete=len(use), k=region_size(grid.n_iter), q=Q_TOP, block=int(block), n_null=int(n_null),
        lift=lift, p_lift=p_lift, region_oos_mean=_mean([r.region_oos_mean for r in use]),
        grid_oos_mean=_mean([r.grid_oos_mean for r in use]),
        oos_positive_share=_mean([r.oos_positive_share for r in use]), precision=precision, p_precision=p_precision,
        ridge_jaccard=_mean([r.ridge_jaccard for r in use]), ridge_shift=_mean([r.ridge_shift for r in use]),
        pct_best_is=_mean([r.pct_best_is for r in use]), pct_best_pooled=_mean([r.pct_best_pooled for r in use]),
        pct_region=_mean([r.pct_region for r in use]), pct_pick=_mean([r.pct_pick for r in use]))
