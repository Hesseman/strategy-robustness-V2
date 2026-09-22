"""Orchestrates parse -> join -> the five tests -> verdicts (spec decision 4) and exports
JSON. Only reason to change: the battery's composition or verdict rules."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from robustness.bars_loader import infer_interval
from robustness.cost_stress import CostStressResult, cost_stress
from robustness.costs import normalize_root, round_trip_usd
from robustness.drawdown import DrawdownResult, drawdown_analysis, margin_check
from robustness.join import Check, join_trades_to_bars
from robustness.null_entry import RandomEntryResult, random_entry_test
from robustness.report_parser import ParsedReport
from robustness.returns import matched_drift_baseline, pct_returns, usd_per_contract
from robustness.temporal import TemporalResult, temporal_test

GATE_ALPHA = 0.05
CAVEAT = ("These tests take the strategy exactly as reported and ask whether its trades beat random "
          "timing, survive costs, and hold up across eras. They do not know how many strategies, "
          "parameter sets or exit rules were tried before this one - there is no multiple-testing "
          "correction. A pass means 'we could not break it with these tests', not 'it works'.")


class ValidationFailed(Exception):
    """The report and bars do not join cleanly; .checks carries the diagnosis."""

    def __init__(self, checks: list[Check]):
        super().__init__("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed and c.severity == "error"))
        self.checks = checks


@dataclass
class BatteryResult:
    meta: dict
    checks: list[Check]
    baseline: RandomEntryResult
    t3: TemporalResult
    t7: CostStressResult
    dd: DrawdownResult
    margin: dict | None
    verdicts: dict
    gates_passed: int
    gates_total: int
    caveat: str


def run_battery(report: ParsedReport, bars: pd.DataFrame, *, n_perm: int = 1000, seed: int = 0,
                today_margin_usd: float | None = None, min_trades: int = 30,
                capital_usd: float | None = None) -> BatteryResult:
    """Run everything: join the report's trades to bars, then the random-entry null (T8a),
    temporal robustness (T3), cost stress (T7) and the drawdown/capital card, and derive a
    verdict per card. Gates read 'insufficient' below min_trades; everything is still
    computed.

    Accepts: report - a ParsedReport (from parse_report); bars - a bar DataFrame (from
    load_bars); n_perm - permutation count for the T8a null; seed - RNG seed for T8a
    (same seed -> same null draw and p-value); today_margin_usd - optional current
    exchange margin in USD; today_margin_usd=None omits the margin line; any float,
    including 0.0, computes it; min_trades - the trade-count floor below which the
    t8a/t7 gate verdicts read 'insufficient' (every test still runs and is reported);
    capital_usd - optional fixed starting capital in USD replacing 5 x CDaR-80 for the
    drawdown card's percentages and the margin line.

    Returns: a BatteryResult carrying the run's meta (symbol/root/interval/trade counts/
    the n_perm and seed used), the join's checks, the raw result of each of the five
    tests, the optional margin dict, a verdict per card, the gate pass/total counts and
    the fixed multiple-testing caveat.

    Guarantees: raises ValidationFailed (carrying joined.checks) when any 'error'-severity
    join check fails - nothing downstream runs in that case; gates_total is always 2
    (t8a, t7) and gates_passed counts only verdicts equal to 'pass' among those two."""
    joined = join_trades_to_bars(report.trades, bars, summary=report.summary, settings=report.settings, min_trades=min_trades)
    if not joined.ok:
        raise ValidationFailed(joined.checks)
    t = joined.trades
    open_ = bars.open.to_numpy()
    holds = np.maximum(t.hold_bars.to_numpy(), 1)
    d = t.direction.to_numpy()
    pct = pct_returns(t)
    usd = usd_per_contract(t, joined.point_value)
    base = matched_drift_baseline(open_, holds, d)
    baseline_usd = base.mean * t.entry_price.to_numpy() * joined.point_value

    t8a = random_entry_test(open_, holds, d, pct, n_perm=n_perm, seed=seed)
    t3 = temporal_test(bars, t.entry_idx.to_numpy(), holds, d, pct, usd, pd.DatetimeIndex(t.exit_time))

    symbol = report.settings.get("symbol") or ""
    ts_cost = float(np.nanmean(2 * (t.comm_side.to_numpy() + t.slip_side.to_numpy())))
    ref = round_trip_usd(symbol)
    if ref is None:
        cost, source = ts_cost, "report"
    else:
        cost, source = ref, "multiwalk"
    t7 = cost_stress(usd, baseline_usd, cost, source, ts_cost_rt_usd=ts_cost)
    dd = drawdown_analysis(usd, pd.DatetimeIndex(t.entry_time), pd.DatetimeIndex(t.exit_time), capital_override=capital_usd)
    margin = margin_check(dd.capital, today_margin_usd) if today_margin_usd is not None else None

    enough = len(t) >= min_trades
    verdicts = {"baseline": "reference",
                "t8a": ("pass" if t8a.p_value < GATE_ALPHA else "fail") if enough else "insufficient",
                "t3": "score",
                "t7": ("pass" if t7.passed_1x else "fail") if enough else "insufficient",
                "drawdown": "reference"}
    meta = {"symbol": symbol, "root": normalize_root(symbol), "interval": report.settings.get("interval"),
            "strategies": report.settings.get("strategies", []), "n_trades": int(len(t)),
            "n_long": int((d > 0).sum()), "n_short": int((d < 0).sum()),
            "first_entry": t.entry_time.min(), "last_exit": t.exit_time.max(),
            "point_value": joined.point_value, "cost_basis": joined.cost_basis,
            "n_bars": int(len(bars)), "bar_interval": str(infer_interval(bars)),
            "bars_start": bars.index[0], "bars_end": bars.index[-1],
            "n_perm": n_perm, "seed": seed, "min_trades": min_trades, "capital_usd": capital_usd,
            "warnings": list(report.warnings)}
    return BatteryResult(meta=meta, checks=joined.checks, baseline=t8a, t3=t3, t7=t7, dd=dd, margin=margin,
                         verdicts=verdicts, gates_passed=sum(1 for k in ("t8a", "t7") if verdicts[k] == "pass"),
                         gates_total=2, caveat=CAVEAT)


def _jsonable(o):
    if o is pd.NaT:
        return None
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _jsonable(getattr(o, f.name)) for f in dataclasses.fields(o)}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict(orient="records"))
    if isinstance(o, (pd.DatetimeIndex, np.ndarray)):
        return [_jsonable(v) for v in o.tolist()] if isinstance(o, pd.DatetimeIndex) else _jsonable(o.tolist())
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def to_json(result: BatteryResult) -> str:
    """Serialize a BatteryResult to JSON for the app/export layer.

    Accepts: a BatteryResult (as returned by run_battery).
    Returns: the whole result as an indented JSON string - nested dataclasses become
    objects (one key per field), dicts and DataFrames (as record lists) recurse the same
    way, NumPy/pandas scalars become plain int/float/bool, non-finite floats (NaN, +-inf)
    and NaT become null, and Timestamps become ISO-8601 strings.
    Guarantees: every top-level BatteryResult field is a key in the output; the result
    round-trips through json.loads into plain dicts/lists/str/float/int/bool/None only;
    every float in the output is finite, so json.dumps(..., allow_nan=False) never raises."""
    return json.dumps(_jsonable(result), indent=1)
