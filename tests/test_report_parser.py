import pandas as pd
import pytest

from robustness.report_parser import ReportFormatError, parse_money, parse_report
from conftest import make_bars, make_trades, trades_to_report_text


def test_parse_money_formats():
    assert parse_money("$59.10") == 59.10
    assert parse_money("($123.40)") == -123.40
    assert parse_money("-0.34%") == -0.34
    assert parse_money("42.34%") == 42.34
    assert parse_money("") != parse_money("")  # nan
    assert parse_money("n/a") != parse_money("n/a")


def test_mini_report_trades(mini_report_text):
    rep = parse_report(mini_report_text)
    t = rep.trades
    assert len(t) == 2
    assert t.trade_id.tolist() == [1, 2]
    assert t.direction.tolist() == [-1, 1]
    assert t.entry_time.tolist() == [pd.Timestamp("2025-01-06 09:00"), pd.Timestamp("2025-01-07 10:00")]
    assert t.exit_time.tolist() == [pd.Timestamp("2025-01-06 11:30"), pd.Timestamp("2025-01-07 13:00")]
    assert t.entry_price.tolist() == [20000.00, 20010.50]
    assert t.exit_price.tolist() == [19990.00, 19998.25]
    assert t.contracts.tolist() == [3, 1]
    assert t.gross_pnl.tolist() == [60.00, -24.50]
    assert t.net_pnl.tolist() == [54.60, -29.90]
    assert t.cum_net_pnl.tolist() == [54.60, 24.70]
    assert t.entry_signal.tolist() == ["ShortEntry A", "LongEntry C"]
    assert t.exit_signal.tolist() == ["CoverExit B", "StopExit D"]
    assert t.comm_side.tolist() == [2.20, 2.20] and t.slip_side.tolist() == [0.50, 0.50]
    assert t.runup_usd.tolist() == [36.00, 16.00] and t.drawdown_usd.tolist() == [-12.00, -28.00]


def test_mini_report_settings_and_summary(mini_report_text):
    rep = parse_report(mini_report_text)
    assert rep.settings["symbol"] == "@MNQ"
    assert rep.settings["interval"] == "30 min."
    assert rep.settings["fixed_contracts"] == "1"
    assert rep.settings["strategies"] == ["Demo_Strategy(On)"]
    assert rep.settings["inputs"] == {"Demo_Strategy - Length": "20", "Demo_Strategy - UseStop": "true"}
    assert rep.summary["Total Number of Trades"] == 2
    assert rep.summary["Avg. Bars in Total Trades"] == 6.5
    assert rep.summary["Total Net Profit"] == 24.70
    assert rep.summary["Percent Profitable"] == 50.0
    assert rep.summary["Trading Period"] == "2 days"      # unparseable -> raw string
    assert rep.warnings == []


def test_synthetic_roundtrip():
    bars = make_bars(n=800, seed=5)
    trades = make_trades(bars, n=30, seed=6)
    rep = parse_report(trades_to_report_text(trades))
    assert len(rep.trades) == len(trades)
    assert rep.trades.entry_time.tolist() == trades.entry_time.tolist()
    assert rep.trades.gross_pnl.tolist() == trades.gross_pnl.tolist()
    assert rep.summary["Total Number of Trades"] == len(trades)


def test_missing_trades_list_header_raises():
    with pytest.raises(ReportFormatError, match="Trades List"):
        parse_report("Performance Summary\nTotal Net Profit,$1.00,,,\n")


def test_scaling_out_is_refused(mini_report_text):
    extra = ",Sell,1/7/2025 13:30,StopExit D,$19999.50,,($22.00),$2.70,,($28.00),30.00%,-10.00%,$2.20,$0.50"
    lines = mini_report_text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(",Sell,1/7/2025 13:00"))
    lines.insert(i + 1, extra)
    with pytest.raises(ReportFormatError, match="trade 2 has 2 exit rows"):
        parse_report("\r\n".join(lines))


def test_exit_type_mismatch_raises(mini_report_text):
    bad = mini_report_text.replace(",Buy to Cover,1/6/2025 11:30", ",Sell,1/6/2025 11:30")
    with pytest.raises(ReportFormatError, match="does not close"):
        parse_report(bad)


def test_trailing_open_trade_is_dropped_with_warning(mini_report_text):
    lines = mini_report_text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(",Sell,1/7/2025 13:00"))
    del lines[i]
    rep = parse_report("\r\n".join(lines))
    assert rep.trades.trade_id.tolist() == [1]
    assert len(rep.warnings) == 1
    w = rep.warnings[0]
    assert "trade 2" in w and "1/7/2025 10:00" in w and "1 of 2 trades used" in w


def test_interior_open_trade_is_still_an_error(mini_report_text):
    lines = mini_report_text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(",Buy to Cover,1/6/2025 11:30"))
    del lines[i]
    with pytest.raises(ReportFormatError, match="trade 1 has no exit row"):
        parse_report("\r\n".join(lines))


def test_only_trade_open_is_an_error(mini_report_text):
    drop = (",Buy to Cover,1/6/2025 11:30", "2,Buy,1/7/2025 10:00", ",Sell,1/7/2025 13:00")
    lines = [l for l in mini_report_text.splitlines() if not l.startswith(drop)]
    with pytest.raises(ReportFormatError, match="still open"):
        parse_report("\r\n".join(lines))


def test_blank_contracts_cell_raises_readable_error(mini_report_text):
    bad = mini_report_text.replace(
        "1,Sell Short,1/6/2025 09:00,ShortEntry A,$20000.00,$0.00,3,$54.60,",
        "1,Sell Short,1/6/2025 09:00,ShortEntry A,$20000.00,$0.00,,$54.60,")
    with pytest.raises(ReportFormatError, match="contracts cell"):
        parse_report(bad)


def test_percent_profit_thousands_comma_does_not_shift_columns(mini_report_text):
    cases = {
        "23,500.00%": [36.00, 16.00],        # thousands comma inside the cell: stripped
        "500.00%": [36.00, 16.00],           # 3-digit percent after a digit-terminated cell: delimiter kept
        "1,234,567.00%": [36.00, 16.00],     # two thousands groups
        "-23,500.00%": [36.00, 16.00],       # negative with a thousands comma: stripped, sign kept
    }
    for pct, runups in cases.items():
        text = mini_report_text.replace(",0.05%,$36.00,", f",{pct},$36.00,")
        assert pct in text
        t = parse_report(text).trades
        assert t.runup_usd.tolist() == runups, pct
        assert t.comm_side.tolist() == [2.20, 2.20] and t.slip_side.tolist() == [0.50, 0.50], pct
