import json

import numpy as np
import pandas as pd
import pytest

from robustness.bars_loader import load_bars
from robustness.battery import GATE_ALPHA, ValidationFailed, run_battery, to_json
from robustness.report_parser import parse_report
from conftest import make_bars, make_trades, bars_to_ts_text, trades_to_report_text


def _battery(planted_edge, n=80, **kw):
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=n, seed=31, planted_edge=planted_edge)
    rep = parse_report(trades_to_report_text(trades))
    return run_battery(rep, load_bars(bars_to_ts_text(bars)), n_perm=500, seed=0, **kw)


def test_trailing_open_trade_is_dropped_and_reported_in_meta():
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    lines = trades_to_report_text(trades).splitlines()
    last_exit = max(k for k, l in enumerate(lines) if l.startswith((",Sell,", ",Buy to Cover,")))
    del lines[last_exit]
    rep = parse_report("\r\n".join(lines))
    r = run_battery(rep, load_bars(bars_to_ts_text(bars)), n_perm=200, seed=0)
    assert r.meta["n_trades"] == 79 and len(rep.trades) == 79
    assert any("79 of 80 trades used" in w for w in r.meta["warnings"])


def _losing_trades(bars, n, seed):
    """make_trades' planted_edge trades with direction inverted, so the timing
    that beat random entry now loses to it - gross/net P&L recomputed to match.
    Point value fixed at 2.0 (MNQ), matching make_trades' own default."""
    t = make_trades(bars, n=n, seed=seed, planted_edge=True)
    t["direction"] = -t["direction"]
    t["gross_pnl"] = (t.direction * (t.exit_price - t.entry_price) * 2.0 * t.contracts).round(2)
    t["net_pnl"] = (t.gross_pnl - 2 * (t.comm_side + t.slip_side)).round(2)
    return t


def test_planted_edge_passes_t8a_gate():
    r = _battery(planted_edge=True)
    assert r.verdicts["t8a"] == "pass" and r.baseline.p_value < GATE_ALPHA
    assert r.verdicts["baseline"] == "reference" and r.verdicts["t3"] == "score" and r.verdicts["drawdown"] == "reference"
    assert r.verdicts["t7"] in ("pass", "fail")
    assert r.gates_total == 2 and r.gates_passed == sum(v == "pass" for k, v in r.verdicts.items() if k in ("t8a", "t7"))
    assert r.meta["symbol"] == "@MNQ" and r.meta["root"] == "MNQ" and r.meta["point_value"] == 2.0
    assert r.t7.cost_source == "multiwalk" and r.t7.cost_rt_usd == 5.74
    assert r.meta["n_trades"] == r.baseline.n_trades == r.dd.n_trades
    assert "multiple-testing" in r.caveat


def test_insufficient_trades_disables_gates():
    r = _battery(planted_edge=True, n=20)
    assert r.verdicts["t8a"] == "insufficient" and r.verdicts["t7"] == "insufficient"
    assert r.gates_passed == 0


def test_unknown_symbol_falls_back_to_report_costs():
    bars = make_bars(n=3000, seed=32)
    trades = make_trades(bars, n=40, seed=33)
    rep = parse_report(trades_to_report_text(trades, symbol="@ZZZ"))
    r = run_battery(rep, load_bars(bars_to_ts_text(bars)), n_perm=200)
    assert r.t7.cost_source == "report" and r.t7.cost_rt_usd == pytest.approx(5.40)


def test_margin_line_optional():
    assert _battery(planted_edge=False, n=40).margin is None
    r = _battery(planted_edge=False, n=40, today_margin_usd=2500.0)
    assert r.margin["today_margin_usd"] == 2500.0 and r.margin["capital_usd"] == r.dd.capital


def test_validation_failure_raises_with_checks():
    bars = make_bars(n=3000, seed=34)
    trades = make_trades(bars, n=40, seed=35)
    rep = parse_report(trades_to_report_text(trades))
    short_bars = load_bars(bars_to_ts_text(bars.iloc[: len(bars) // 2]))
    with pytest.raises(ValidationFailed) as ei:
        run_battery(rep, short_bars, n_perm=100)
    assert any(c.name == "timestamps_found" and not c.passed for c in ei.value.checks)


def test_to_json_roundtrips():
    r = _battery(planted_edge=True, n=40)
    d = json.loads(to_json(r))
    assert set(d) >= {"meta", "checks", "baseline", "t3", "t7", "dd", "verdicts", "gates_passed", "gates_total", "caveat"}
    assert d["baseline"]["n_perm"] == 500 and len(d["baseline"]["null"]) == 500
    assert isinstance(d["dd"]["worst"]["peak_time"], str)
    assert d["t3"]["yearly"][0].keys() >= {"year", "usd", "n"}


def test_to_json_handles_nat_and_meta_timestamps():
    from robustness.battery import _jsonable
    assert _jsonable({"t": pd.NaT, "x": float("nan")}) == {"t": None, "x": None}
    d = json.loads(to_json(_battery(planted_edge=True, n=40)))
    for key in ("bars_start", "bars_end", "first_entry", "last_exit"):
        assert isinstance(d["meta"][key], str)


def test_inverted_edge_fails_both_gates():
    bars = make_bars(n=4000, seed=30)
    trades = _losing_trades(bars, n=80, seed=31)
    rep = parse_report(trades_to_report_text(trades))
    r = run_battery(rep, load_bars(bars_to_ts_text(bars)), n_perm=500, seed=0)
    assert r.verdicts["t8a"] == "fail"
    assert r.verdicts["t7"] == "fail"
    assert r.gates_passed == 0
    assert r.baseline.p_value > 0.5


def test_jsonable_writes_infinite_values_as_null():
    from robustness.battery import _jsonable
    assert _jsonable({"a": float("inf"), "b": np.float64("-inf"), "c": float("nan")}) == {"a": None, "b": None, "c": None}
    json.dumps(_jsonable(_battery(planted_edge=True, n=40)), allow_nan=False)


def test_zero_margin_is_a_value_not_absence():
    r = _battery(planted_edge=False, n=40, today_margin_usd=0.0)
    assert r.margin is not None
    assert r.margin["today_margin_usd"] == 0.0 and r.margin["margin_to_equity"] == 0.0


def test_battery_capital_usd_reaches_the_drawdown_card_and_margin():
    r = _battery(planted_edge=True, capital_usd=25_000.0, today_margin_usd=2_000.0)
    assert r.dd.capital_basis == "fixed" and r.dd.capital == 25_000.0 and r.meta["capital_usd"] == 25_000.0
    assert r.margin["contracts_covered"] == pytest.approx(25_000.0 / 2_000.0)
    r0 = _battery(planted_edge=True)
    assert r0.dd.capital_basis == "5 x CDaR-80" and r0.meta["capital_usd"] is None
