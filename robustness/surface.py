"""Per-iteration performance over a date window from daily mark-to-market $ P&L (spec decision 4).
Equity starts flat at 0; drawdown = equity - running peak (<= 0); avg DD = mean over all days
(reproduces MultiWalk's 'Avg DD' within 1%); NP/AvgDD = net / -avg DD, NaN when never in drawdown."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robustness.multiwalk_text import MultiWalkGrid


@dataclass
class SurfaceMetrics:
    net_profit: np.ndarray
    max_dd: np.ndarray
    avg_dd: np.ndarray
    np_over_avgdd: np.ndarray
    n_trades: np.ndarray
    n_days: int


def daily_drawdown_stats(pnl: np.ndarray) -> tuple[float, float, float]:
    """Accepts a 1-D daily $ P&L series. Returns (net_profit, max_dd, avg_dd) where equity is the
    cumulative sum from a flat 0 start, drawdown = equity - running peak, max_dd = its minimum and
    avg_dd = its mean over all days. Guarantees max_dd <= avg_dd <= 0 and (0, 0, 0) for empty input."""
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0, 0.0, 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = eq - peak
    return float(eq[-1]), float(dd.min()), float(dd.mean())


def window_metrics(grid: MultiWalkGrid, mask: np.ndarray) -> SurfaceMetrics:
    """Vectorised daily_drawdown_stats for every iteration over the days where mask is True,
    plus the number of closed trades whose exit date falls inside the masked date range.
    Accepts: a MultiWalkGrid and a boolean mask over grid.dates. Returns: SurfaceMetrics with
    one value per iteration. Guarantees: np_over_avgdd is NaN where avg_dd == 0; an empty mask
    gives zeros and n_days == 0."""
    mask = np.asarray(mask, dtype=bool)
    n = grid.n_iter
    n_days = int(mask.sum())
    if n_days == 0:
        z = np.zeros(n)
        return SurfaceMetrics(z, z.copy(), z.copy(), np.full(n, np.nan), np.zeros(n, dtype=int), 0)
    sub = grid.daily_pnl[:, mask]
    eq = np.cumsum(sub, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = eq - peak
    net, max_dd, avg_dd = eq[:, -1], dd.min(axis=1), dd.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(avg_dd < 0, net / -avg_dd, np.nan)
    lo = np.datetime64(grid.dates[mask][0].date()); hi = np.datetime64(grid.dates[mask][-1].date())
    n_trades = np.array([int(((d.astype("datetime64[D]") >= lo) & (d.astype("datetime64[D]") <= hi)).sum()) for d in grid.exit_dates], dtype=int)
    return SurfaceMetrics(net_profit=net, max_dd=max_dd, avg_dd=avg_dd, np_over_avgdd=ratio, n_trades=n_trades, n_days=n_days)


def metric_values(m: SurfaceMetrics, metric: str) -> np.ndarray:
    """'NPAvgDD' -> m.np_over_avgdd, 'NP' -> m.net_profit; any other name raises ValueError."""
    if metric == "NPAvgDD":
        return m.np_over_avgdd
    if metric == "NP":
        return m.net_profit
    raise ValueError(f"unsupported metric {metric!r} (NPAvgDD or NP)")
