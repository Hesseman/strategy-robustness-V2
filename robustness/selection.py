"""Selection haircut (spec decision 7): is the best of the grid, or MultiWalk's pick, better than
the best that pure luck produces across the same grid? White (2000) Reality Check with the
stationary bootstrap (Politis & Romano 1994) on the in-sample daily P&L - the max-over-scan null
of signal_lab/robustness/t8b_deflation.py applied to a strategy grid. Reference only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid
from robustness.windows import Window


@dataclass
class SelectionWindow:
    index: int
    label: str
    complete: bool
    n_iter: int
    n_days: int
    n_eff: float
    best_index: int
    best_mean: float
    pick_index: int
    pick_mean: float
    p_best: float
    p_pick: float
    null_max: np.ndarray
    insufficient: bool = False


@dataclass
class SelectionResult:
    windows: list[SelectionWindow]
    n_boot: int
    block_len: float


def stationary_bootstrap_indices(n: int, rng: np.random.Generator, mean_block: float) -> np.ndarray:
    """Politis-Romano stationary bootstrap: a resample of n day indices built from blocks whose
    length is geometric with mean mean_block, wrapping around the end. Returns int array (n,)."""
    p = 1.0 / max(1.0, mean_block)
    restart = rng.random(n) < p
    starts = rng.integers(0, n, size=n)
    idx = np.empty(n, dtype=int)
    idx[0] = starts[0]
    for t in range(1, n):
        idx[t] = starts[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def effective_trials(sub: np.ndarray) -> float:
    """Participation ratio (sum lambda)^2 / sum lambda^2 of the eigenvalues of the correlation matrix
    of the rows of sub (iterations x days); constant rows are dropped. Guarantees 1 <= result <=
    number of non-constant rows, 0.0 when none, and the row count when only one remains."""
    s = np.asarray(sub, dtype=float)
    s = s[s.std(axis=1) > 0]
    if s.shape[0] == 0:
        return 0.0
    if s.shape[0] == 1:
        return 1.0
    lam = np.linalg.eigvalsh(np.corrcoef(s))
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def reality_check(sub: np.ndarray, pick_index: int, *, n_boot: int = 500, mean_block: float = 20.0, seed: int = 0) -> SelectionWindow:
    """Reality Check on one window's IS daily P&L (iterations x days).
    Statistic per iteration = mean daily $; null_max[b] = max over iterations of the re-centred
    bootstrap mean; p_best = (1 + #{null_max >= best mean}) / (1 + n_boot), p_pick likewise for
    pick_index. Returns a SelectionWindow with index/label/complete left at 0/''/True for the
    caller to fill. Deterministic for a given seed."""
    sub = np.asarray(sub, dtype=float)
    n_iter, n_days = sub.shape
    f = sub.mean(axis=1)
    centered = sub - f[:, None]
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n_days, rng, mean_block)
        null_max[b] = centered[:, idx].mean(axis=1).max()
    best = int(np.argmax(f))
    p_best = float((1 + int((null_max >= f[best]).sum())) / (1 + n_boot))
    p_pick = float((1 + int((null_max >= f[pick_index]).sum())) / (1 + n_boot))
    return SelectionWindow(index=0, label="", complete=True, n_iter=n_iter, n_days=n_days, n_eff=effective_trials(sub),
                           best_index=best, best_mean=float(f[best]), pick_index=int(pick_index), pick_mean=float(f[pick_index]),
                           p_best=p_best, p_pick=p_pick, null_max=null_max)


def selection_test(grid: MultiWalkGrid, windows: list[Window], *, n_boot: int = 500, mean_block: float = 20.0,
                   seed: int = 0, min_days: int = 60) -> SelectionResult:
    """Reality Check on every complete window's IS days. Windows that are incomplete or have fewer
    than min_days IS days get an insufficient placeholder (NaN p-values, empty null)."""
    out: list[SelectionWindow] = []
    for w in windows:
        if not w.complete or w.n_is_days < min_days:
            out.append(SelectionWindow(index=w.index, label=w.label, complete=w.complete, n_iter=grid.n_iter, n_days=w.n_is_days,
                                       n_eff=float("nan"), best_index=-1, best_mean=float("nan"), pick_index=w.grid_row - 1,
                                       pick_mean=float("nan"), p_best=float("nan"), p_pick=float("nan"), null_max=np.empty(0),
                                       insufficient=True))
            continue
        r = reality_check(grid.daily_pnl[:, w.is_mask], w.grid_row - 1, n_boot=n_boot, mean_block=mean_block, seed=seed + w.index)
        r.index, r.label, r.complete = w.index, w.label, w.complete
        out.append(r)
    return SelectionResult(windows=out, n_boot=n_boot, block_len=mean_block)
