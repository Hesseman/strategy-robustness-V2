"""Headless render of the timing-sensitivity app through streamlit.testing.v1.AppTest, on
synthetic files. Proves the script runs end to end (parse -> join -> both delay curves ->
cards) without a browser."""
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from robustness import charts
from robustness.bars_loader import load_bars
from robustness.report_parser import parse_report
from robustness.synthetic import bars_to_ts_text, make_bars, make_trades, trades_to_report_text
from robustness.timing import run_timing

APP = str(Path(__file__).resolve().parents[1] / "app" / "timing_app.py")


def _write_sample(tmp_path):
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    report, bars_file = tmp_path / "report.csv", tmp_path / "bars.txt"
    report.write_bytes(trades_to_report_text(trades).encode("utf-8"))
    bars_file.write_bytes(bars_to_ts_text(bars).encode("utf-8"))
    return report, bars_file


def _markdown(at) -> str:
    return "\n".join(el.value for el in at.markdown)


def test_timing_app_renders_without_files():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Upload both files" in el.value for el in at.info)


def test_timing_app_renders_on_sample(tmp_path, monkeypatch):
    report, bars_file = _write_sample(tmp_path)
    monkeypatch.setenv("SR_SAMPLE_REPORT", str(report))
    monkeypatch.setenv("SR_SAMPLE_BARS", str(bars_file))
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _markdown(at)
    assert "Entry delay" in text and "Exit delay" in text
    assert "n alive at k = 10" in text and "100% of entries were filled at the bar open" in text
    assert re.search(r"(?<!\\)\$\d", text) is None, "an unescaped dollar amount reached st.markdown (renders as LaTeX)"
    assert "\\$" in text
    assert len(at.get("plotly_chart")) == 2
    assert "Where timing matters" in text and "hindsight: entering 1 bar earlier" in text
    assert "hindsight: exiting 1 bar earlier" in text and "best shift in -10..+10" in text


def test_timing_app_renders_demo_and_mode_change():
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["demo"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _markdown(at)
    assert "Entry delay" in text and "Exit delay" in text
    at.radio[0].set_value("fixed_hold").run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Entry delay" in _markdown(at)
    at.slider[0].set_value(4).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "n alive at k = 4" in _markdown(at)


def test_fig_delay_curve_has_one_point_per_delay():
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    r = run_timing(parse_report(trades_to_report_text(trades)), load_bars(bars_to_ts_text(bars)), max_k=7)
    for c in (r.entry, r.exit):
        fig = charts.fig_delay_curve(c, "1 contract, gross")
        assert len(fig.data[0].x) == 8 and list(fig.data[0].x) == list(range(8))
        assert fig.data[0].marker.color[0] == charts.ACCENT and fig.data[1].yaxis == "y2"


def test_fig_delay_curve_with_earlier_side():
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    r = run_timing(parse_report(trades_to_report_text(trades)), load_bars(bars_to_ts_text(bars)), max_k=5)
    fig = charts.fig_delay_curve(r.entry, "1 contract, gross", early=r.entry_early)
    assert list(fig.data[0].x) == list(range(-5, 6))
    assert fig.data[0].y[5] == r.entry.points[0].total_usd and fig.data[0].y[0] == r.entry_early.points[5].total_usd
    assert fig.data[0].marker.symbol[0] == "circle-open" and fig.data[0].marker.symbol[6] == "circle"
    assert len(fig.data[1].x) == 11
