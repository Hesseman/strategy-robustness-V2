"""The recommended base setting (docs/wfc-region-lift.md, 'Base setting'): once the WFC verdict is
in, which parameter combination to trade. Kaufman's rule - take the best settings and trade the
middle of them, not the peak - done on the grid: the basis is the last walk-forward window's in-
sample plus its out-of-sample to date (the later half of the history on the 1-split scheme); the
region is the largest connected top-20% area of the pooled net-profit surface over that basis.
The pick follows the verdict (user decision 2026-09-26): after an edge, the pooled peak - on the
real edges tested the strongest part of the region held best out of sample - with the region's
centre (region_wfc.centre_pick) and a spread ensemble (region_wfc.spread_ensemble) as the
diversified alternative; otherwise the centre. The confidence: 'supported' after an edge, 'low
stakes' on a plateau (the choice hardly matters - take the centre), 'none' otherwise. The
literal average of the five best combinations is reported for display, flagged when those five
are not one connected area; it is never the pick. Parameters are classified by name (dollar or
point amounts vs bars, percentages, multiples) for labelling only: on the real grids tested,
dollar and point optima did not follow volatility, so nothing is rescaled. Only reason to change:
the recommendation rule."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from robustness.multiwalk_text import MultiWalkGrid
from robustness.region_wfc import centre_pick, largest_component, neighbourhoods, pool, spread_ensemble, top_set
from robustness.surface import window_metrics
from robustness.windows import Window

KAUFMAN_TOP = 5                                    # the "five best" of Kaufman's display line; fixed
CONFIDENCE = {"edge": "supported", "plateau": "low stakes"}      # any other reading: "none"
PICK_RULE = {"edge": "pooled peak"}                              # any other reading: "centre"

# Name fragments, matched case-insensitively. Scale-free words win, so 'StopATR' and
# 'ProfitTargetMultiplier' stay scale-free; a name with neither reads 'unclassified'.
_SCALE_FREE = ("pct", "percent", "bar", "len", "period", "lookback", "atr", "mult", "ratio", "factor", "day", "hold",
               "rsi", "ema", "sma", "adx", "time", "thresh")
_SCALE_FREE_WORDS = {"fac"}
_PRICE = ("amt", "amount", "dollar", "usd", "$", "pts", "point", "stop", "target", "targ", "tgt", "profit", "dol")
_PRICE_WORDS = {"sl", "tp"}


def classify_axis(name: str) -> str:
    """Accepts a parameter name. Returns 'scale-free' (bars, lengths, percentages, ATR multiples,
    ratios, times), 'price-scaled' (dollar or point amounts, stops, targets) or 'unclassified'.
    Guarantees scale-free words win over price words; a label only, never used to rescale."""
    low = name.lower()
    words = {w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)}
    if any(s in low for s in _SCALE_FREE) or words & _SCALE_FREE_WORDS:
        return "scale-free"
    if any(s in low for s in _PRICE) or words & _PRICE_WORDS:
        return "price-scaled"
    return "unclassified"


def basis_mask(windows: list[Window], n_days: int, scheme: str) -> np.ndarray:
    """The days the base setting is fitted on. Accepts the windows in schedule order, the number
    of trading days and the window scheme. Returns a boolean mask: the last window's in-sample plus
    its out-of-sample to the last data day; on the 'single' scheme (one window spanning all the
    history), or without windows, the later half of the days. Guarantees a non-empty mask."""
    if scheme == "single" or not windows:
        mask = np.zeros(n_days, dtype=bool)
        mask[n_days // 2:] = True
        return mask
    last = windows[-1]
    return np.asarray(last.is_mask, dtype=bool) | np.asarray(last.oos_mask, dtype=bool)


@dataclass
class BaseSetting:
    confidence: str                  # 'supported' | 'low stakes' | 'none'
    reading: str                     # the WFC verdict reading it follows
    basis_start: pd.Timestamp
    basis_end: pd.Timestamp
    n_basis_days: int
    pick: int                        # iteration index of the base setting
    pick_rule: str                   # 'pooled peak' after an edge, else 'centre'
    pick_params: list[float]
    peak: int                        # the pooled surface's best combination (ties: grid order)
    peak_params: list[float]
    peak_in_region: bool             # the peak lies inside the region (it can top a smaller area)
    centre: int                      # the region's centre (region_wfc.centre_pick)
    centre_is_peak: bool             # the region had < 3 cells: the pooled peak stands in as its centre
    centre_params: list[float]
    ensemble: list[int]              # centre first; weight 1/len each
    ensemble_params: list[list[float]]
    ensemble_spacing: int            # 2 inside a region with interior cells, 1 on a thin ridge
    ensemble_interior: bool
    component: np.ndarray            # the region: largest connected top-20% component (boolean mask)
    n_component: int
    ranges: list[tuple[float, float]]    # per parameter, the region's lowest and highest value
    axis_class: list[str]            # per parameter, classify_axis()
    kaufman: int                     # grid combination nearest the mean position of the five best
    kaufman_contiguous: bool         # those five form one connected area
    x: np.ndarray                    # net profit over the basis per combination
    pooled: np.ndarray               # its neighbourhood means


def base_setting(grid: MultiWalkGrid, windows: list[Window], reading: str, scheme: str) -> BaseSetting:
    """The recommended base setting.

    Accepts the grid, its windows, the WFC verdict reading ('edge' | 'loser' | 'plateau' |
    'noise' | 'insufficient') and the window scheme. Returns a BaseSetting computed over
    basis_mask(): the centre of the pooled surface's ridge (or its peak on a ridge under 3 cells),
    the pooled peak, the pick (the peak after an edge, else the centre), the spread ensemble, the
    region's size and parameter ranges, the parameter classes, and Kaufman's five-best average for
    display. Guarantees everything is computed whatever the reading - the confidence says whether
    to use it - and deterministic output."""
    mask = basis_mask(windows, grid.n_days, scheme)
    x = window_metrics(grid, mask).net_profit.astype(float)
    pos = np.asarray(grid.grid_pos)
    pooled = pool(x, neighbourhoods(pos))
    centre = centre_pick(pooled, pos)
    ens = spread_ensemble(centre.component, pos, pooled, centre.index)
    comp = centre.component
    peak = int(top_set(pooled, 1)[0])
    rule = PICK_RULE.get(reading, "centre")
    pick = peak if rule == "pooled peak" else centre.index
    top = top_set(x, min(KAUFMAN_TOP, grid.n_iter))
    mean_pos = pos[top].mean(axis=0)
    kaufman = int(np.argmin(np.linalg.norm(pos - mean_pos, axis=1)))
    top_mask = np.zeros(grid.n_iter, dtype=bool)
    top_mask[top] = True
    dates = grid.dates[mask]
    return BaseSetting(
        confidence=CONFIDENCE.get(reading, "none"), reading=reading, basis_start=pd.Timestamp(dates[0]),
        basis_end=pd.Timestamp(dates[-1]), n_basis_days=int(mask.sum()), pick=pick, pick_rule=rule,
        pick_params=[float(v) for v in grid.params[pick]], peak=peak, peak_params=[float(v) for v in grid.params[peak]],
        peak_in_region=bool(comp[peak]), centre=centre.index,
        centre_is_peak=centre.is_peak, centre_params=[float(v) for v in grid.params[centre.index]],
        ensemble=ens.indices, ensemble_params=[[float(v) for v in grid.params[i]] for i in ens.indices],
        ensemble_spacing=ens.spacing, ensemble_interior=ens.interior, component=comp, n_component=int(comp.sum()),
        ranges=[(float(grid.params[comp, a].min()), float(grid.params[comp, a].max())) for a in range(pos.shape[1])],
        axis_class=[classify_axis(n) for n in grid.param_names], kaufman=kaufman,
        kaufman_contiguous=bool(largest_component(top_mask, pos).sum() == top.size), x=x, pooled=pooled)
