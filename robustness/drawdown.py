"""Drawdown & capital card (spec decision 3). Equity = cumulative 1-contract $ P&L ordered
by trade close. Discrete drawdown episodes (peak -> trough -> recovery, empyrical/pyfolio
get_top_drawdowns convention), CDaR-beta = mean of the worst (1-beta) share of episode
depths, capital = capital_mult x CDaR-80, annual % = annual $ / capital. Sharpe / Sortino /
Calmar / Profit-per-avg-DD are ranking metrics only. drawdown_episodes, cdar, sharpe and
sortino are ported from le-trading-research/scratch/es_drawdown_episodes.py. The margin
line is the present-day-only check from TODO.md (no historical-constancy claim)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def drawdown_episodes(pnl: np.ndarray) -> list[dict]:
    """Identify drawdown episodes from cumulative P&L.

    Accepts: 1D array of per-trade P&L (USD or pct), any numeric dtype.

    Returns: list of dicts, one per episode. Each dict has keys:
      - peak_i: index where equity last peaked before the drawdown; -1 means the
        drawdown is already underway at the flat start (equity never exceeded 0
        before this drawdown) - the caller maps -1 to a time before the first trade
      - trough_i: index where equity reached its lowest point in this drawdown
      - recovery_i: index where equity recovered to the peak (None if unrecovered)
      - depth: absolute drawdown depth (always <= 0)

    Guarantees: depth <= 0; episodes are non-overlapping and chronological;
      a flat 0.0 start point precedes trade 0, so a losing run beginning at the
      first trade is counted instead of being invisible against an initial peak
      of its own making; cumsum(pnl)[trough_i] - cumsum(pnl)[peak_i] == depth for
      peak_i >= 0, and -cumsum(pnl)[trough_i] == depth (peak equity was the flat
      0.0 start) when peak_i == -1.
    """
    padded = np.concatenate([[0.0], np.cumsum(np.asarray(pnl, dtype=float))])
    running_max = np.maximum.accumulate(padded)
    underwater = padded - running_max
    episodes: list[dict] = []
    in_dd = False; peak_i = 0; trough_i = 0; trough_val = 0.0
    for i in range(len(padded)):
        if underwater[i] >= 0:
            if in_dd:
                episodes.append({"peak_i": peak_i - 1, "trough_i": trough_i - 1, "recovery_i": i - 1, "depth": float(trough_val)})
                in_dd = False
            peak_i = i
        else:
            if not in_dd:
                in_dd, trough_i, trough_val = True, i, underwater[i]
            elif underwater[i] < trough_val:
                trough_i, trough_val = i, underwater[i]
    if in_dd:
        episodes.append({"peak_i": peak_i - 1, "trough_i": trough_i - 1, "recovery_i": None, "depth": float(trough_val)})
    return episodes


def cdar(depths_positive: np.ndarray, beta: float = 0.80) -> float:
    """Conditional drawdown at risk: mean of the worst (1-beta) share of episode depths.

    Accepts: 1D array of episode depths as positive numbers (e.g., 120.0 for -120 depth),
      beta in (0, 1) controlling the quantile cutoff (0.80 = worst 20%).

    Returns: float, the mean of the worst k episodes where k = ceil((1-beta) * N).
      Returns 0.0 if the array is empty.

    Guarantees: result >= 0; always includes at least one episode if N > 0.
    """
    d = np.sort(np.asarray(depths_positive, dtype=float))
    if d.size == 0:
        return 0.0
    k = max(1, int(math.ceil((1.0 - beta) * d.size)))
    return float(d[-k:].mean())


def sharpe(pnl: np.ndarray, years: float) -> float:
    """Per-trade Sharpe ratio, annualized by the number of trades per year.

    Accepts: 1D array of per-trade P&L values, years as float > 0 (time span of trades).

    Returns: float; annualized Sharpe = (mean(pnl) / std(pnl)) * sqrt(trades / years).
      Returns 0.0 if std is zero or std > 0 cannot be computed.

    Guarantees: uses ddof=1 for unbiased std; returns 0.0 (not inf or NaN) when denominator is 0.
    """
    pnl = np.asarray(pnl, dtype=float)
    std = pnl.std(ddof=1) if len(pnl) > 1 else 0.0
    return float(pnl.mean() / std * np.sqrt(len(pnl) / years)) if std > 0 else 0.0


def sortino(pnl: np.ndarray, years: float) -> float:
    """Per-trade Sortino ratio with MAR=0, annualized like Sharpe.

    Accepts: 1D array of per-trade P&L values, years as float > 0.

    Returns: float; annualized Sortino = (mean(pnl) / downside_dev) * sqrt(trades / years),
      where downside_dev = sqrt(mean(min(pnl, 0)^2)). Returns 0.0 if downside_dev is zero.

    Guarantees: only negative returns contribute to the denominator (MAR=0);
      returns 0.0 (not inf or NaN) when downside_dev is 0.
    """
    pnl = np.asarray(pnl, dtype=float)
    dd = float(np.sqrt(np.mean(np.minimum(pnl, 0.0) ** 2)))
    return float(pnl.mean() / dd * np.sqrt(len(pnl) / years)) if dd > 0 else 0.0


@dataclass
class DrawdownResult:
    n_trades: int
    years: float
    sum_usd: float
    annual_usd: float
    equity: np.ndarray
    times: pd.DatetimeIndex
    episodes: list[dict]
    depths: np.ndarray           # positive, descending
    n_episodes: int
    n_unrecovered: int
    max_dd: float
    mean_dd: float
    median_dd: float
    cdar80: float
    capital_mult: float
    capital: float
    capital_basis: str
    capital_5x_cdar80: float
    annual_pct: float
    max_dd_pct_of_capital: float
    sharpe: float
    sortino: float
    calmar: float
    profit_over_avg_dd: float
    profit_over_cdar80: float
    dur_peak_trough_days: dict   # {"mean":..,"max":..}
    dur_peak_recovery_days: dict
    worst: dict                  # peak_time, trough_time, recovery_time|None, depth


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b and not math.isnan(b) else float("nan")


def drawdown_analysis(usd_pc: np.ndarray, entry_times: pd.DatetimeIndex, exit_times: pd.DatetimeIndex,
                      capital_mult: float = 5.0, beta: float = 0.80,
                      capital_override: float | None = None) -> DrawdownResult:
    """Compute drawdown episodes and all ranking/capital metrics on a trade sequence.

    Accepts: 1D array of per-trade P&L in USD (1 contract), entry_times and exit_times
      as DatetimeIndex (length must match usd_pc), capital_mult (scaling factor for CDaR-beta),
      beta (quantile level, default 0.80). capital_override - when given (must be > 0) it
      replaces capital_mult x CDaR-beta as the capital used for annual_pct and
      max_dd_pct_of_capital; capital_basis reads "fixed" then, else "5 x CDaR-80";
      capital_5x_cdar80 always carries capital_mult x CDaR-beta. Raises ValueError for
      capital_override <= 0.

    Returns: DrawdownResult dataclass with all 24 fields populated:
      - Trade counts, time spans, P&L totals
      - Equity curve (cumsum of P&L), times, episodes, depths
      - Risk metrics (max_dd, mean_dd, median_dd, cdar80)
      - Capital sizing (capital_mult, capital, capital_basis, capital_5x_cdar80, annual_pct,
        max_dd_pct_of_capital)
      - Ranking metrics (sharpe, sortino, calmar, profit_over_avg_dd, profit_over_cdar80)
      - Duration metrics (peak-to-trough, peak-to-recovery, mean and max days)
      - Worst episode detail (peak_time, trough_time, recovery_time or None, depth)

    Guarantees: trades are sorted by exit_time; years = max(1 day, last_exit - first_entry);
      equity is in chronological order of trade closes; worst episode is the deepest drawdown;
      all divisions return NaN (not inf) when denominator is 0 or NaN.
    """
    usd_pc = np.asarray(usd_pc, dtype=float)
    exit_times = pd.DatetimeIndex(exit_times); entry_times = pd.DatetimeIndex(entry_times)
    order = np.argsort(exit_times.to_numpy(), kind="stable")
    pnl = usd_pc[order]; times = exit_times[order]

    def _t(i: int):
        """Episode-index -> timestamp, honouring the flat-start convention.

        Accepts: i, an episode's peak_i/trough_i/recovery_i (-1 or a valid
          position into `times`).

        Returns: times[i] for i >= 0; entry_times.min() for i == -1, since the
          flat start precedes every trade's own entry.
        """
        return times[i] if i >= 0 else entry_times.min()

    years = max((exit_times.max() - entry_times.min()).days, 1) / 365.25
    equity = np.cumsum(pnl)
    eps = drawdown_episodes(pnl)
    depths = np.sort(np.array([abs(e["depth"]) for e in eps], dtype=float))[::-1]
    max_dd = float(depths[0]) if depths.size else 0.0
    mean_dd = float(depths.mean()) if depths.size else 0.0
    median_dd = float(np.median(depths)) if depths.size else 0.0
    c80 = cdar(depths, beta)
    capital_5x = capital_mult * c80
    if capital_override is not None:
        if not capital_override > 0:
            raise ValueError("capital_override must be > 0")
        capital, basis = float(capital_override), "fixed"
    else:
        capital, basis = float(capital_5x), "5 x CDaR-80"
    annual_usd = float(pnl.sum() / years)
    pt = [(_t(e["trough_i"]) - _t(e["peak_i"])).days for e in eps]
    pr = [(_t(e["recovery_i"]) - _t(e["peak_i"])).days for e in eps if e["recovery_i"] is not None]
    worst_e = min(eps, key=lambda e: e["depth"]) if eps else None
    worst = ({"peak_time": _t(worst_e["peak_i"]), "trough_time": _t(worst_e["trough_i"]),
              "recovery_time": _t(worst_e["recovery_i"]) if worst_e["recovery_i"] is not None else None,
              "depth": worst_e["depth"]} if worst_e else {"peak_time": None, "trough_time": None, "recovery_time": None, "depth": 0.0})
    return DrawdownResult(
        n_trades=len(pnl), years=float(years), sum_usd=float(pnl.sum()), annual_usd=annual_usd,
        equity=equity, times=times, episodes=eps, depths=depths, n_episodes=len(eps),
        n_unrecovered=sum(1 for e in eps if e["recovery_i"] is None),
        max_dd=max_dd, mean_dd=mean_dd, median_dd=median_dd, cdar80=c80, capital_mult=capital_mult,
        capital=float(capital), capital_basis=basis, capital_5x_cdar80=float(capital_5x),
        annual_pct=_safe_div(annual_usd, capital),
        max_dd_pct_of_capital=_safe_div(max_dd, capital),
        sharpe=sharpe(pnl, years), sortino=sortino(pnl, years), calmar=_safe_div(annual_usd, max_dd),
        profit_over_avg_dd=_safe_div(annual_usd, mean_dd), profit_over_cdar80=_safe_div(annual_usd, c80),
        dur_peak_trough_days={"mean": float(np.mean(pt)) if pt else 0.0, "max": float(max(pt)) if pt else 0.0},
        dur_peak_recovery_days={"mean": float(np.mean(pr)) if pr else 0.0, "max": float(max(pr)) if pr else 0.0},
        worst=worst)


def margin_check(capital_usd: float, today_margin_usd: float) -> dict:
    """Check current margin requirement against CDaR-80 capital (deployment feasibility line).

    Accepts: capital_usd (float, CDaR-80 capital from drawdown_analysis),
      today_margin_usd (float, current exchange margin requirement).

    Returns: dict with keys:
      - today_margin_usd: input margin (as float)
      - capital_usd: input capital (as float)
      - margin_to_equity: ratio of margin to capital (NaN if capital is 0)
      - contracts_covered: how many contracts the capital can support (NaN if margin is 0)

    Guarantees: uses NaN (not inf) for division by zero; makes no claim about historical
      margin levels, only the present-day calculation.
    """
    return {"today_margin_usd": float(today_margin_usd), "capital_usd": float(capital_usd),
            "margin_to_equity": _safe_div(today_margin_usd, capital_usd),
            "contracts_covered": _safe_div(capital_usd, today_margin_usd)}
