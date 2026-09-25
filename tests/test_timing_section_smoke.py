"""Headless render of the main page's timing-sensitivity section through
streamlit.testing.v1.AppTest, on synthetic files: the section appears under the five cards,
reuses their uploads, reacts to its own controls and never stops the MultiWalk section below."""
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from robustness.synthetic import bars_to_ts_text, make_bars, make_trades, trades_to_report_text

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _write(tmp_path, bars_slice=slice(None)):
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    report, bars_file = tmp_path / "report.csv", tmp_path / "bars.txt"
    report.write_bytes(trades_to_report_text(trades).encode("utf-8"))
    bars_file.write_bytes(bars_to_ts_text(bars.iloc[bars_slice]).encode("utf-8"))
    return report, bars_file


def _markdown(at) -> str:
    return "\n".join(el.value for el in at.markdown)


def _headers(at) -> list[str]:
    return [h.value for h in at.header]


def _run_sample(tmp_path, monkeypatch, bars_slice=slice(None)):
    report, bars_file = _write(tmp_path, bars_slice)
    monkeypatch.setenv("SR_SAMPLE_REPORT", str(report))
    monkeypatch.setenv("SR_SAMPLE_BARS", str(bars_file))
    at = AppTest.from_file(APP, default_timeout=240)
    at.session_state["sample"] = True
    return at.run()


def test_no_timing_section_without_files():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert not any("Timing sensitivity" in h for h in _headers(at))


def test_timing_section_renders_under_the_cards_and_above_multiwalk(tmp_path, monkeypatch):
    at = _run_sample(tmp_path, monkeypatch)
    assert not at.exception, [str(e) for e in at.exception]
    heads = _headers(at)
    timing = next(i for i, h in enumerate(heads) if "Timing sensitivity" in h)
    multiwalk = next(i for i, h in enumerate(heads) if "MultiWalk" in h)
    assert timing < multiwalk
    blocks = [el.value for el in at.markdown]
    drawdown = next(i for i, v in enumerate(blocks) if v.startswith("### Drawdown"))
    entry = next(i for i, v in enumerate(blocks) if v.startswith("### Entry delay"))
    assert drawdown < entry, "the timing cards must sit under the five cards"
    text = _markdown(at)
    assert "Gates passed" in text
    assert "Entry delay" in text and "Exit delay" in text
    assert "n alive at k = 10" in text and "100% of entries were filled at the bar open" in text
    assert "Where timing matters" in text and "hindsight: entering 1 bar earlier" in text
    assert "hindsight: exiting 1 bar earlier" in text and "best shift" not in text
    assert "per trade on the" in text and "that still fit change by" in text
    assert re.search(r"(?<!\\)\$\d", text) is None, "an unescaped dollar amount reached st.markdown"
    specs = [c.proto.spec for c in at.get("plotly_chart")]
    assert sum("entry shift" in s for s in specs) == 1 and sum("exit shift" in s for s in specs) == 1


def test_timing_controls_rerender(tmp_path, monkeypatch):
    at = _run_sample(tmp_path, monkeypatch)
    at.radio(key="timing_mode").set_value("fixed_hold").run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _markdown(at)
    assert "hindsight: shifting the whole trade 1 bar earlier" in text and "delaying the whole trade 1 bar" in text
    assert "hindsight: exiting 1 bar earlier" in text, "the exit card does not depend on the mode"
    assert "Where timing matters" in text, "the leg comparison always uses single-leg curves"
    assert any("the exit moves with the entry" in c.value for c in at.caption)
    at.slider(key="timing_max_k").set_value(4).run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _markdown(at)
    assert "n alive at k = 4" in text and "per trade on the" in text


def test_timing_controls_keep_their_values_while_the_section_is_hidden(tmp_path, monkeypatch):
    at = _run_sample(tmp_path, monkeypatch)
    at.slider(key="timing_max_k").set_value(4).run()
    at.radio(key="timing_mode").set_value("fixed_hold").run()
    at.session_state["sample"] = False
    at.run()
    assert not any("Timing sensitivity" in h for h in _headers(at))
    at.session_state["sample"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="timing_max_k").value == 4 and at.radio(key="timing_mode").value == "fixed_hold"
    assert "n alive at k = 4" in _markdown(at)


def test_a_failure_inside_the_section_is_a_warning_and_multiwalk_still_renders(tmp_path, monkeypatch):
    from app import timing_section

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(timing_section, "where_lines", boom)
    at = _run_sample(tmp_path, monkeypatch)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Timing section skipped: RuntimeError: boom" in el.value for el in at.warning)
    assert any("MultiWalk" in h for h in _headers(at))


def test_no_timing_section_when_the_files_do_not_join(tmp_path, monkeypatch):
    at = _run_sample(tmp_path, monkeypatch, bars_slice=slice(0, 2000))
    assert not at.exception, [str(e) for e in at.exception]
    assert any("do not join" in el.value for el in at.error)
    heads = _headers(at)
    assert not any("Timing sensitivity" in h for h in heads)
    assert any("MultiWalk" in h for h in heads)
