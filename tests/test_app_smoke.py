"""Headless render of the Streamlit app through streamlit.testing.v1.AppTest, on
synthetic files. Proves the script runs end to end (parse -> battery -> cards)
without a browser; the browser-pane check is the controller's job."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from robustness.synthetic import bars_to_ts_text, make_bars, make_trades, trades_to_report_text

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _write_sample(tmp_path):
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    report, bars_file = tmp_path / "report.csv", tmp_path / "bars.txt"
    report.write_bytes(trades_to_report_text(trades).encode("utf-8"))
    bars_file.write_bytes(bars_to_ts_text(bars).encode("utf-8"))
    return report, bars_file


def _write_losing_sample(tmp_path):
    """Same planted-edge trades as _write_sample with direction inverted, so the
    timing that beat random entry now loses to it (mirrors test_battery's
    _losing_trades; point value fixed at 2.0, matching make_trades' default)."""
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    trades["direction"] = -trades["direction"]
    trades["gross_pnl"] = (trades.direction * (trades.exit_price - trades.entry_price) * 2.0 * trades.contracts).round(2)
    trades["net_pnl"] = (trades.gross_pnl - 2 * (trades.comm_side + trades.slip_side)).round(2)
    report, bars_file = tmp_path / "report.csv", tmp_path / "bars.txt"
    report.write_bytes(trades_to_report_text(trades).encode("utf-8"))
    bars_file.write_bytes(bars_to_ts_text(bars).encode("utf-8"))
    return report, bars_file


def test_app_renders_full_battery_on_sample(tmp_path, monkeypatch):
    report, bars_file = _write_sample(tmp_path)
    monkeypatch.setenv("SR_SAMPLE_REPORT", str(report))
    monkeypatch.setenv("SR_SAMPLE_BARS", str(bars_file))
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(el.value for el in at.markdown)
    assert "Gates passed" in text
    assert "REFERENCE" in text and ("PASS" in text or "FAIL" in text)
    assert "random sets matched or beat the strategy" in text
    assert "p = 0.000" not in text, "T8a p must never read 0.000 ((k+1)/(n+1) with 'p ≤' when k = 0)"
    import re
    joined = "\n".join(el.value for el in at.markdown)
    assert re.search(r"(?<!\\)\$\d", joined) is None, "an unescaped dollar amount reached st.markdown (renders as LaTeX)"
    assert "\\$" in joined


def test_app_renders_fail_pills(tmp_path, monkeypatch):
    report, bars_file = _write_losing_sample(tmp_path)
    monkeypatch.setenv("SR_SAMPLE_REPORT", str(report))
    monkeypatch.setenv("SR_SAMPLE_BARS", str(bars_file))
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    joined = "\n".join(el.value for el in at.markdown)
    assert "FAIL" in joined
    assert "Gates passed: 0 of 2" in joined


def test_app_renders_demo():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["demo"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Gates passed" in " ".join(el.value for el in at.markdown)


def _write_mw_sample(tmp_path):
    from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
    text, sched = make_multiwalk({"A": list(range(5)), "B": list(range(4))}, n_days=600, seed=8, structure="persistent", split=400)
    d = tmp_path / "mw"; (d / "Optimization Files").mkdir(parents=True); (d / "Walkforward Files").mkdir()
    (d / "Optimization Files" / "Synthetic_MW [@SYN-30min]_MultiWalk.txt").write_text(text, encoding="utf-8")
    (d / "Walkforward Files" / "WalkforwardData.db").write_bytes(make_walkforward_db(sched))
    return d


def test_app_renders_without_files_and_shows_both_sections():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [str(e) for e in at.exception]
    text = " ".join(el.value for el in at.markdown) + " ".join(el.value for el in at.info)
    assert "MultiWalk surface tests" in " ".join(h.value for h in at.header)
    assert "Upload both files" in text


def test_app_renders_multiwalk_section_on_sample(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_SAMPLE_MW_DIR", str(_write_mw_sample(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["mw_sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    joined = "\n".join(el.value for el in at.markdown)
    assert "Walk Forward Correlation" in joined and "Plateau" in joined and "Selection haircut" in joined
    assert "✓ PASS" in joined and "region lift pooled over" in joined and "Tinsley's correlation (continuity, not the gate)" in joined
    tabs = [t.label for t in at.tabs]
    assert tabs[:3] == ["Surface", "Scatter", "Ranked profile"] and "Bands" not in tabs   # 20 combinations: no bands
    assert "Null (lift)" in tabs and "Null (ρ)" in tabs
    assert "Read with the PASS:" in joined and "ahead of the grid in **1 of 1** complete window" in joined
    assert "context, not a gate" in joined
    assert "Base setting - which combination to trade?" in joined and "SUPPORTED" in joined
    assert "**base setting:** A = " in joined and "centre of a region of" in joined and "Kaufman's average" in joined
    assert "the centre rule window by window" in joined
    assert "top 6 in-sample → median out-of-sample rank" in joined
    import re
    assert re.search(r"(?<!\\)\$\d", joined) is None, "an unescaped dollar amount reached st.markdown"


def test_multiwalk_window_choice_rerenders_with_trade_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("SR_SAMPLE_MW_DIR", str(_write_mw_sample(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["mw_sample"] = True
    at.run()
    assert "windows: MultiWalk's windows" in " ".join(c.value for c in at.caption)
    for scheme, label in (("single", "split: IS"), ("two", "window 2: IS")):
        at.radio(key="mw_windows").set_value(scheme).run()
        assert not at.exception, [str(e) for e in at.exception]
        joined = "\n".join(el.value for el in at.markdown)
        assert label in joined and "trades per combination in/out" in joined and "best in-sample" in joined


def test_multiwalk_plateau_branch_renders(tmp_path, monkeypatch):
    """Identical variants - every combination carries iteration 1's daily P&L and trades - leave
    one distinct out-of-sample pattern, fewer than twice the region: not scored, plateau."""
    from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
    text, sched = make_multiwalk({"A": list(range(5)), "B": list(range(4))}, n_days=600, seed=8, structure="noise", split=400)
    head, first, *rest = text.strip().split("\n")
    body = first.split("~", 1)[1]
    text = "\n".join([head, first] + [line.split("~", 1)[0] + "~" + body for line in rest]) + "\n"
    d = tmp_path / "mw_noise"; (d / "Optimization Files").mkdir(parents=True); (d / "Walkforward Files").mkdir()
    (d / "Optimization Files" / "Noise_MW [@SYN-30min]_MultiWalk.txt").write_text(text, encoding="utf-8")
    (d / "Walkforward Files" / "WalkforwardData.db").write_bytes(make_walkforward_db(sched))
    monkeypatch.setenv("SR_SAMPLE_MW_DIR", str(d))
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["mw_sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    joined = "\n".join(el.value for el in at.markdown)
    assert "WFC gate not applied" in joined and "PLATEAU - parameter choice immaterial, gate not applied" in joined
    assert "repeat out of sample" in joined and "Gates passed: 0 of 1" not in joined
    assert "LOW STAKES - take the centre" in joined


def test_capital_radio_switches_the_drawdown_line(tmp_path, monkeypatch):
    report, bars_file = _write_sample(tmp_path)
    monkeypatch.setenv("SR_SAMPLE_REPORT", str(report)); monkeypatch.setenv("SR_SAMPLE_BARS", str(bars_file))
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["sample"] = True
    at.run()
    at.radio(key="capital_basis").set_value("Fixed starting capital").run()
    joined = "\n".join(el.value for el in at.markdown)
    assert "your starting capital" in joined and "5 x CDaR-80 would be" in joined
