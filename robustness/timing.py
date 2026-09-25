"""Timing sensitivity: how much of the return survives when a trade's entry or exit is
acted on k bars late. Two experiments, each a curve over k = 0..max_k:

- Delayed entry: the entry moves to the open of bar entry_idx + k. In the default
  'fixed_exit' mode the exit stays as reported (its bar and its actual fill price): we cannot
  re-run the strategy's stop/target logic, so the exit is treated as a time-fixed signal. A
  trade whose delayed entry reaches or passes its exit bar (entry_idx + k >= exit_idx) is
  skipped at that k - nobody enters after the exit signal. In 'fixed_hold' mode both legs
  move k bars (entry at open[entry_idx + k], exit at open[exit_idx + k]), so the hold length
  is preserved; a trade is skipped when its moved exit falls past the last bar.
- Delayed exit: the entry stays as reported; the exit moves to the open of bar exit_idx + k.
  A trade is skipped when exit_idx + k > n_bars - 1. Identical in both modes.

Earlier (hindsight, reference only): the same two experiments run the other way. An
earlier entry fills at open[entry_idx - k] and is skipped when that bar is before the first
bar ('fixed_hold' moves the exit k bars earlier with it); an earlier exit fills at
open[exit_idx - k] and is skipped when it would reach the entry bar (exit_idx - k <=
entry_idx). No strategy can act before its signal fires, so these curves are not tradable:
they show how much of the move the signal lags, i.e. whether work on an earlier entry
trigger or an earlier exit trigger has more to gain.

k = 0 is the strategy exactly as reported: actual fill prices for both legs. A moved leg
always fills at the open of the bar it is moved to (TradeStation 'next bar at market').
Trades are independent: a delayed exit may overlap the next entry and no position limit is
modelled. Returns are gross, one contract, at the join's inferred point value; costs are
constant per trade, so they shift the curve's level, not its shape. Deterministic - no
randomness. Skipped trades are excluded from every metric and counted per k.

Only reason to change: the delay conventions above or the per-k metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from robustness.bars_loader import infer_interval
from robustness.battery import ValidationFailed, _jsonable
from robustness.costs import normalize_root
from robustness.join import Check, join_trades_to_bars
from robustness.report_parser import ParsedReport

LEGS = ("entry", "exit")
MODES = ("fixed_exit", "fixed_hold")
CAVEAT = ("This is a picture of fragility, not a test with a verdict. A moved leg fills at the open of the bar "
          "it moves to; in the default mode the exit keeps its reported bar and price because the strategy's own "
          "stop/target logic cannot be re-run on the shifted position. Trades are treated as independent (a "
          "delayed exit may overlap the next entry) and returns are gross, one contract, without costs - costs "
          "move the curve's level, not its shape. The earlier side of each curve (negative shift) is hindsight: no "
          "strategy can act before its signal fires, so it is a reference for where an earlier trigger would pay, "
          "not a tradable result.")


@dataclass
class DelayPoint:
    k: int
    n_alive: int
    n_skipped: int
    total_usd: float
    mean_usd: float
    mean_pct: float
    win_rate: float
    profit_factor: float
    retention: float


@dataclass
class DelayCurve:
    """kind: which leg moves ('entry' | 'exit'); mode: 'fixed_exit' | 'fixed_hold';
    points: one DelayPoint per k = 0..max_k; first_nonpositive_k: smallest k >= 1 with at
    least one alive trade and total_usd <= 0, else None; per_trade_pct: n_trades x
    (max_k + 1) % returns, NaN where the trade is skipped; shift: 'later' (a delay, the
    tradable direction) | 'earlier' (hindsight, reference only) - k counts bars in that
    direction."""
    kind: str
    mode: str
    points: list[DelayPoint]
    first_nonpositive_k: int | None
    per_trade_pct: np.ndarray
    shift: str = "later"


@dataclass
class TimingResult:
    meta: dict
    checks: list[Check]
    entry: DelayCurve
    exit: DelayCurve
    entry_early: DelayCurve
    exit_early: DelayCurve
    at_open_share: float
    caveat: str


def _check_leg_mode(leg: str, mode: str) -> None:
    if leg not in LEGS:
        raise ValueError(f"leg must be one of {LEGS}, got {leg!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def delayed_returns(trades: pd.DataFrame, open_: np.ndarray, k: int, leg: str,
                    mode: str = "fixed_exit", early: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move one leg of every trade k bars later (or, with early=True, k bars earlier -
    hindsight, reference only) and price the result.

    Accepts: trades - joined trades (direction, entry_idx, exit_idx, entry_price,
    exit_price); open_ - the bar opens the indices point into; k - delay in bars, int >= 0;
    leg - 'entry' | 'exit'; mode - 'fixed_exit' | 'fixed_hold' (only changes the entry leg);
    early - move the leg earlier instead of later.
    Returns: (pct, usd_points, alive) per trade - pct = direction x (exit / entry - 1),
    usd_points = direction x (exit - entry) in price points (multiply by the point value for
    $ per contract), alive = the trade still fits at this k. pct and usd_points are NaN where
    alive is False.
    Guarantees: k = 0 uses the actual fills for both legs, so it reproduces
    returns.pct_returns and usd_per_contract / point_value exactly with every trade alive; a
    moved leg fills at open_[idx + k] (open_[idx - k] when early); a later entry is skipped
    once it reaches the exit bar, an earlier exit once it reaches the entry bar, and any
    moved leg outside the bar series is skipped; the input is not mutated; raises ValueError on a bad
    k, leg or mode."""
    _check_leg_mode(leg, mode)
    if not _is_int(k) or k < 0:
        raise ValueError(f"k must be an int >= 0, got {k!r}")
    open_ = np.asarray(open_, dtype=float)
    last = len(open_) - 1
    d = trades.direction.to_numpy(dtype=float)
    e = trades.entry_idx.to_numpy(dtype=int)
    x = trades.exit_idx.to_numpy(dtype=int)
    ep = trades.entry_price.to_numpy(dtype=float)
    xp = trades.exit_price.to_numpy(dtype=float)

    if k == 0:
        alive, entry, exit_ = np.ones(len(d), dtype=bool), ep, xp
    elif early and leg == "exit":
        alive = x - k > e
        entry, exit_ = ep, open_[np.maximum(x - k, 0)]
    elif early and mode == "fixed_exit":
        alive = e - k >= 0
        entry, exit_ = open_[np.maximum(e - k, 0)], xp
    elif early:  # fixed_hold: both legs move earlier, hold preserved
        alive = e - k >= 0
        entry, exit_ = open_[np.maximum(e - k, 0)], open_[np.maximum(x - k, 0)]
    elif leg == "exit":
        alive = x + k <= last
        entry, exit_ = ep, open_[np.minimum(x + k, last)]
    elif mode == "fixed_exit":
        alive = e + k < x
        entry, exit_ = open_[np.minimum(e + k, last)], xp
    else:  # fixed_hold: both legs move, hold preserved
        alive = x + k <= last
        entry, exit_ = open_[np.minimum(e + k, last)], open_[np.minimum(x + k, last)]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(alive, d * (exit_ / entry - 1.0), np.nan)
        pts = np.where(alive, d * (exit_ - entry), np.nan)
    return pct, pts, alive


def _point(k: int, pct: np.ndarray, usd: np.ndarray, alive: np.ndarray, base_total: float | None) -> DelayPoint:
    p, u = pct[alive], usd[alive]
    n = int(alive.sum())
    total = float(u.sum()) if n else 0.0
    if n:
        wins, losses = float(u[u > 0].sum()), float(-u[u < 0].sum())
        pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else float("nan"))
    else:
        pf = float("nan")
    if base_total is None:  # this is the k = 0 point
        base_total = total
    return DelayPoint(k=k, n_alive=n, n_skipped=int(len(alive) - n), total_usd=total,
                      mean_usd=float(u.mean()) if n else float("nan"),
                      mean_pct=float(p.mean()) if n else float("nan"),
                      win_rate=float((u > 0).mean()) if n else float("nan"),
                      profit_factor=pf,
                      retention=total / base_total if base_total > 0 else float("nan"))


def delay_curve(trades: pd.DataFrame, open_: np.ndarray, point_value: float, leg: str,
                mode: str = "fixed_exit", max_k: int = 10, early: bool = False) -> DelayCurve:
    """Aggregate delayed_returns over k = 0..max_k (bars later, or earlier when early=True).

    Accepts: trades and open_ as for delayed_returns; point_value - $ per 1.0 price point
    per contract; leg - 'entry' | 'exit'; mode - 'fixed_exit' | 'fixed_hold'; max_k - int >= 1.
    Returns: a DelayCurve with one DelayPoint per k: n_alive / n_skipped, total_usd (sum of
    one-contract gross $ over alive trades, 0.0 when none is alive), mean_usd, mean_pct,
    win_rate, profit_factor (inf when there are wins and no losses) and retention =
    total_usd[k] / total_usd[0].
    Guarantees: mean_usd / mean_pct / win_rate / profit_factor are NaN where no trade is
    alive; retention is NaN at every k when total_usd[0] <= 0 (never a misleading ratio);
    first_nonpositive_k ignores k with no alive trade, so running out of trades does not
    read as the profit turning; raises ValueError on a bad leg, mode or max_k."""
    _check_leg_mode(leg, mode)
    if not _is_int(max_k) or max_k < 1:
        raise ValueError(f"max_k must be an int >= 1, got {max_k!r}")
    per_trade = np.full((len(trades), max_k + 1), np.nan)
    points: list[DelayPoint] = []
    base_total = None
    for k in range(max_k + 1):
        pct, pts, alive = delayed_returns(trades, open_, k, leg, mode, early)
        per_trade[:, k] = pct
        pt = _point(k, pct, pts * point_value, alive, base_total)
        if k == 0:
            base_total = pt.total_usd
        points.append(pt)
    first = next((p.k for p in points[1:] if p.n_alive > 0 and p.total_usd <= 0), None)
    return DelayCurve(kind=leg, mode=mode, points=points, first_nonpositive_k=first, per_trade_pct=per_trade,
                      shift="earlier" if early else "later")


def run_timing(report: ParsedReport, bars: pd.DataFrame, *, max_k: int = 10, mode: str = "fixed_exit",
               min_trades: int = 30) -> TimingResult:
    """Join the report's trades to the bars and compute both delay curves.

    Accepts: report - a ParsedReport (from parse_report); bars - a bar DataFrame (from
    load_bars); max_k - the largest delay in bars, int >= 1; mode - 'fixed_exit' (default:
    a delayed entry keeps the reported exit) | 'fixed_hold' (a delayed entry moves the exit
    too); min_trades - passed to the join's (warn-only) trade-count check.
    Returns: a TimingResult with the run's meta (symbol/root/interval/trade counts/point
    value/bars/max_k/mode/warnings), the join's checks, the entry-delay and exit-delay
    curves, the matching earlier (hindsight) curves, the share of entries filled at the bar
    open, and the fixed caveat.
    Guarantees: raises ValidationFailed (carrying the join's checks) when any
    'error'-severity join check fails - nothing is computed in that case; raises
    ValueError on a bad max_k or mode before joining; deterministic; the report's trade
    table is not mutated."""
    if not _is_int(max_k) or max_k < 1:
        raise ValueError(f"max_k must be an int >= 1, got {max_k!r}")
    _check_leg_mode("entry", mode)
    joined = join_trades_to_bars(report.trades, bars, summary=report.summary, settings=report.settings,
                                 min_trades=min_trades)
    if not joined.ok:
        raise ValidationFailed(joined.checks)
    t = joined.trades
    open_ = bars.open.to_numpy(dtype=float)
    pv = joined.point_value
    d = t.direction.to_numpy()
    at_open = float(np.mean(np.isclose(t.entry_price.to_numpy(dtype=float), open_[t.entry_idx.to_numpy()])))
    symbol = report.settings.get("symbol") or ""
    meta = {"symbol": symbol, "root": normalize_root(symbol), "interval": report.settings.get("interval"),
            "strategies": report.settings.get("strategies", []), "n_trades": int(len(t)),
            "n_long": int((d > 0).sum()), "n_short": int((d < 0).sum()),
            "first_entry": t.entry_time.min(), "last_exit": t.exit_time.max(),
            "point_value": pv, "cost_basis": joined.cost_basis,
            "n_bars": int(len(bars)), "bar_interval": str(infer_interval(bars)),
            "bars_start": bars.index[0], "bars_end": bars.index[-1],
            "max_k": int(max_k), "mode": mode, "warnings": list(report.warnings)}
    return TimingResult(meta=meta, checks=joined.checks,
                        entry=delay_curve(t, open_, pv, "entry", mode, int(max_k)),
                        exit=delay_curve(t, open_, pv, "exit", mode, int(max_k)),
                        entry_early=delay_curve(t, open_, pv, "entry", mode, int(max_k), early=True),
                        exit_early=delay_curve(t, open_, pv, "exit", mode, int(max_k), early=True),
                        at_open_share=at_open, caveat=CAVEAT)


def to_json(result: TimingResult) -> str:
    """Serialize a TimingResult to JSON for the app's download button.

    Accepts: a TimingResult (as returned by run_timing).
    Returns: the whole result as an indented JSON string, with battery.to_json's rules
    (nested dataclasses become objects, NumPy/pandas scalars become plain values,
    Timestamps become ISO-8601 strings).
    Guarantees: every top-level TimingResult field is a key in the output; every non-finite
    float (NaN for skipped trades and empty points, inf profit factor) is null, so
    json.dumps(..., allow_nan=False) on the parsed output never raises."""
    return json.dumps(_jsonable(result), indent=1)
