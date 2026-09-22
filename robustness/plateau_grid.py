"""Plateau test on the optimisation grid (port of signal_lab/robustness/t1_plateau.py; spec
decision 6). Neighbourhood = the pick plus every combination within one step on every axis.
score = 0.5 * flat + 0.5 * positive share; flat = 1 - min(1, std/mean) when the neighbourhood
mean > 0 else 0; positive share = neighbours with metric > 0. Scored on the OOS metric around the
IS pick; the IS score is reported for comparison. A score, never a gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid
from robustness.windows import Window


@dataclass
class PlateauWindow:
    index: int
    label: str
    complete: bool
    pick_index: int
    neighbour_index: np.ndarray
    is_values: np.ndarray       # pick first, then neighbours (NaN kept)
    oos_values: np.ndarray
    score_oos: float
    flat_oos: float
    positive_share_oos: float
    score_is: float
    n_neighbours: int


@dataclass
class PlateauResult:
    windows: list[PlateauWindow]
    pooled_score: float
    n_complete: int


def neighbours(grid_pos: np.ndarray, i: int) -> np.ndarray:
    """Indices of the combinations at Chebyshev distance exactly 1 from combination i on the grid."""
    d = np.abs(grid_pos - grid_pos[i]).max(axis=1)
    return np.flatnonzero(d == 1)


def plateau_score(values: np.ndarray) -> tuple[float, float, float]:
    """Accepts the neighbourhood's metric values, pick first. Returns (score, flat, positive_share)
    per the module rule, using finite values only. Guarantees all three in [0, 1] and
    (NaN, NaN, NaN) when fewer than 2 finite values - too little data to score, never a spike."""
    v = np.asarray(values, dtype=float)
    f = v[np.isfinite(v)]
    if f.size < 2:
        return float("nan"), float("nan"), float("nan")
    mean = float(f.mean())
    flat = max(0.0, 1.0 - min(1.0, float(f.std()) / (abs(mean) + 1e-9))) if mean > 0 else 0.0
    nb = v[1:]
    nb = nb[np.isfinite(nb)]
    share = float((nb > 0).mean()) if nb.size else 0.0
    return float(max(0.0, min(1.0, 0.5 * flat + 0.5 * share))), flat, share


def plateau_test(pairs: list[tuple[np.ndarray, np.ndarray]], windows: list[Window], grid: MultiWalkGrid) -> PlateauResult:
    """Plateau score per window and pooled over complete windows.
    Accepts: pairs[i] = (IS metric, OOS metric) per iteration for windows[i]; the grid.
    Returns: PlateauResult; pooled_score = mean score_oos over the complete windows that could be
    scored (NaN when none of them could)."""
    out: list[PlateauWindow] = []
    for (x, y), w in zip(pairs, windows):
        i = w.grid_row - 1
        nb = neighbours(grid.grid_pos, i)
        is_vals = np.concatenate([[x[i]], x[nb]]); oos_vals = np.concatenate([[y[i]], y[nb]])
        s_oos, flat, share = plateau_score(oos_vals)
        s_is, _, _ = plateau_score(is_vals)
        out.append(PlateauWindow(index=w.index, label=w.label, complete=w.complete, pick_index=i, neighbour_index=nb,
                                 is_values=is_vals, oos_values=oos_vals, score_oos=s_oos, flat_oos=flat,
                                 positive_share_oos=share, score_is=s_is, n_neighbours=int(len(nb))))
    complete = np.array([p.score_oos for p in out if p.complete], dtype=float)
    pooled = float(np.nanmean(complete)) if np.isfinite(complete).any() else float("nan")
    return PlateauResult(windows=out, pooled_score=pooled, n_complete=int(complete.size))
