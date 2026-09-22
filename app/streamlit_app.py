"""Strategy Robustness App - Streamlit UI. Uploads -> parse -> validate -> battery -> cards."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.copy import CARDS  # noqa: E402
from robustness import charts  # noqa: E402
from robustness.bars_loader import BarsFormatError, load_bars  # noqa: E402
from robustness.battery import ValidationFailed, run_battery, to_json  # noqa: E402
from robustness.report_parser import ReportFormatError, parse_report  # noqa: E402

st.set_page_config(page_title="Strategy Robustness V2", layout="wide")

PILL = {"pass": ("#2e8b57", "✓ PASS"), "fail": ("#c0392b", "✗ FAIL"), "score": ("#1f5fbf", "SCORE"),
        "reference": ("#9a9a94", "REFERENCE"), "insufficient": ("#9a9a94", "n < 30 - gate not applied")}


def pill(verdict: str, extra: str = "") -> str:
    """Accepts a verdict key (pass | fail | score | reference | insufficient) and optional
    extra text; returns the HTML span for the coloured pill. Guarantees the five known
    keys render; any other key raises KeyError."""
    color, label = PILL[verdict]
    return (f'<span style="background:{color};color:white;padding:4px 12px;border-radius:14px;'
            f'font-weight:600;font-size:0.9rem">{label}{(" " + extra) if extra else ""}</span>')


def _ts_metric(summary: dict, key: str, kind: str) -> str:
    v = summary.get(key)
    if not isinstance(v, (int, float)):
        return "-"
    return {"money": f"${v:,.2f}", "ratio": f"{v:.2f}", "pct": f"{v:.1f}%", "int": f"{int(v)}"}[kind]


def _interval_label(value: str) -> str:
    """Accepts the str(Timedelta) the battery stores (e.g. '0 days 00:30:00'); returns '30 min',
    '1 h', '1 day' or the input unchanged when it does not parse. Guarantees no exception."""
    try:
        td = pd.Timedelta(value)
    except Exception:
        return value
    minutes = int(td.total_seconds() // 60)
    if minutes % 1440 == 0:
        return f"{minutes // 1440} day" + ("s" if minutes // 1440 != 1 else "")
    if minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


@st.cache_data(show_spinner=False)
def _parse(report_bytes: bytes):
    return parse_report(report_bytes.decode("utf-8-sig", errors="replace"))


@st.cache_data(show_spinner=False)
def _bars(bars_bytes: bytes):
    return load_bars(bars_bytes.decode("utf-8-sig", errors="replace"))


@st.cache_data(show_spinner=False)
def demo_files() -> tuple[bytes, bytes]:
    """Synthetic report + bars (planted edge, seeds 30/31) for the demo button."""
    from robustness.synthetic import bars_to_ts_text, make_bars, make_trades, trades_to_report_text
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, planted_edge=True)
    return trades_to_report_text(trades).encode("utf-8"), bars_to_ts_text(bars).encode("utf-8")


@st.cache_data(show_spinner="Running the battery...")
def _battery(report_bytes: bytes, bars_bytes: bytes, n_perm: int, seed: int, margin: float | None):
    return run_battery(_parse(report_bytes), _bars(bars_bytes), n_perm=n_perm, seed=seed, today_margin_usd=margin)


def _p_text(k_ge: int, p: float) -> str:
    """T8a p for display. p = (k+1)/(n+1) is a bound when no random set matched (k = 0), so
    it is shown as 'p ≤' rounded up; otherwise 'p =' to 3 decimals, 4 when 3 would hide
    which side of the 0.05 gate it is on."""
    if k_ge == 0:
        return f"p ≤ {math.ceil(p * 1000) / 1000:.3f}"
    if p < 0.001 or (round(p, 3) >= 0.05 > p):
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


def card(key: str, verdict: str, result_lines: list[str], fig: go.Figure,
         fig2: go.Figure | None = None, extra: str = "") -> None:
    """Accepts a key from CARDS, its verdict, the result bullet lines and one or two
    plotly figures; renders one bordered card (copy and pill left, charts right).
    Returns nothing; guarantees the card order and layout are identical for every test."""
    c = CARDS[key]
    with st.container(border=True):
        st.markdown(f"### {c['title']}")
        st.caption(c["tagline"])
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(f"**What it catches.** {c['catches']}")
            st.markdown(f"**How it works.** {c['how']}".replace("$", chr(92) + "$"))
            st.markdown("**Result**")
            for line in result_lines:
                st.markdown(f"- {line.replace('$', chr(92) + '$')}")
            st.markdown(pill(verdict, extra), unsafe_allow_html=True)
        with right:
            st.plotly_chart(fig, width="stretch")
            if fig2 is not None:
                st.plotly_chart(fig2, width="stretch")


st.title("Strategy Robustness V2")
st.write("Upload a TradeStation **Strategy Performance Report** (saved as CSV) and the **bar data** it ran on "
         "(Data Window export, same symbol and interval). Nothing is stored.")

with st.sidebar:
    st.header("Inputs")
    up_report = st.file_uploader("Performance report (.csv)", type=["csv", "txt"])
    up_bars = st.file_uploader("Bar data (.txt / .csv)", type=["txt", "csv"])
    n_perm = st.select_slider("Random-entry sets (T8a)", options=[500, 1000, 2000, 5000], value=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)
    margin_in = st.number_input("Today's margin per contract, $ (optional)", min_value=0.0, value=0.0, step=50.0)
    sample_report, sample_bars = os.environ.get("SR_SAMPLE_REPORT"), os.environ.get("SR_SAMPLE_BARS")
    use_sample = False
    if sample_report and sample_bars and Path(sample_report).exists() and Path(sample_bars).exists():
        use_sample = st.button("Load sample files (dev)")
        if use_sample:
            st.session_state["sample"] = True
    if st.button("Try the demo (synthetic strategy)"):
        st.session_state["demo"] = True
    if st.session_state.get("demo") or st.session_state.get("sample"):
        if st.button("Clear loaded data"):
            st.session_state.pop("demo", None); st.session_state.pop("sample", None); st.rerun()
    if st.session_state.get("demo"):
        report_bytes, bars_bytes = demo_files()
        st.caption("synthetic demo - a planted timing edge on random-walk bars, not a real strategy")
    elif st.session_state.get("sample"):
        report_bytes, bars_bytes = Path(sample_report).read_bytes(), Path(sample_bars).read_bytes()
        st.caption(f"sample: {Path(sample_report).name} + {Path(sample_bars).name}")
    else:
        report_bytes = up_report.getvalue() if up_report else None
        bars_bytes = up_bars.getvalue() if up_bars else None

if not (report_bytes and bars_bytes):
    st.info("Upload both files to start.")
    st.stop()

try:
    result = _battery(report_bytes, bars_bytes, int(n_perm), int(seed), float(margin_in) or None)
except ReportFormatError as e:
    st.error(f"Report not readable: {e}"); st.stop()
except BarsFormatError as e:
    st.error(f"Bar file not readable: {e}"); st.stop()
except ValidationFailed as e:
    st.error("The report and the bars do not join - fix the inputs before any test can run.")
    st.table([{"check": c.name, "ok": "✓" if c.passed else "✗", "detail": c.detail} for c in e.checks])
    st.stop()
except Exception as e:
    st.error("Could not process these files - check that the report is the CSV Strategy Performance "
             "Report and the bars are a Data Window export of the same symbol and interval. Details: "
             + f"{type(e).__name__}: {e}")
    st.stop()

m = result.meta
st.subheader(f"{m['symbol']} · {m['interval']} · {m['n_trades']} trades ({m['n_long']} long / {m['n_short']} short) "
             f"· {m['first_entry']:%Y-%m-%d} → {m['last_exit']:%Y-%m-%d}")
st.caption((f"strategies: {', '.join(m['strategies']) or '-'} · point value ${m['point_value']:g} (inferred) · "
           f"{m['n_bars']:,} bars at {_interval_label(m['bar_interval'])} · {m['n_perm']} random sets, seed {m['seed']}"
           ).replace("$", chr(92) + "$"))
for w in m["warnings"]:
    st.warning(w)

result_summary = _parse(report_bytes).summary
st.caption("TradeStation's own summary (as traded, its cost assumptions). The cards below are per contract with our cost model.")
strip = st.columns(6)
for col, (label, key, kind) in zip(strip, [
        ("Net profit", "Total Net Profit", "money"), ("Profit factor", "Profit Factor", "ratio"),
        ("% profitable", "Percent Profitable", "pct"), ("Trades", "Total Number of Trades", "int"),
        ("Avg trade net", "Avg. Trade Net Profit", "money"), ("Win:loss ratio", "Ratio Avg. Win:Avg. Loss", "ratio")]):
    col.metric(label, _ts_metric(result_summary, key, kind))

st.markdown(f"**Gates passed: {result.gates_passed} of {result.gates_total}.** " + result.caveat)
with st.expander("Validation checks", expanded=not all(c.passed for c in result.checks)):
    st.table([{"check": c.name, "ok": "✓" if c.passed else ("⚠" if c.severity == "warn" else "✗"),
               "detail": c.detail} for c in result.checks])

b, t3, t7, dd = result.baseline, result.t3, result.t7, result.dd
card("baseline", result.verdicts["baseline"],
     [f"strategy mean return **{b.observed_mean*100:+.3f}%** per trade (1 contract, % of price), win rate {b.observed_win_rate:.0%}",
      f"random entry with the same holds and directions **{b.baseline_mean*100:+.3f}%**, win rate {b.baseline_win_rate:.0%}",
      f"lift **{b.lift*100:+.3f} pp** per trade"],
     charts.fig_baseline(b))
card("t8a", result.verdicts["t8a"],
     [f"strategy **{b.observed_mean*100:+.3f}%** vs random-set mean {b.null_mean*100:+.3f}% (sd {b.null_std*100:.3f}%) → z = {b.z:.1f}",
      f"**{b.k_ge} of {b.n_perm}** random sets matched or beat the strategy → {_p_text(b.k_ge, b.p_value)} (gate < 0.05)",
      f"best random set {b.null.max()*100:+.3f}%"],
     charts.fig_null_hist(b))
crisis = t3.crisis
crisis_line = (f"crisis bars (top-decile volatility): lift **{crisis['lift_crisis']*100:+.3f}%** (n={crisis['n_crisis']}) vs "
               f"{crisis['lift_calm']*100:+.3f}% elsewhere (n={crisis['n_calm']})"
               if crisis["lift_crisis"] is not None and crisis["lift_calm"] is not None
               else f"crisis split: n={crisis['n_crisis']} crisis / {crisis['n_calm']} calm trades - too few for a lift in at least one side")
card("t3", result.verdicts["t3"],
     [f"**{t3.windows_positive} of {t3.windows_total}** windows with positive lift: " +
      ", ".join(f"{(w.lift or 0)*100:+.2f}%" if w.counted else f"(n={w.n})" for w in t3.windows),
      crisis_line,
      f"**{t3.years_positive} of {t3.years_total}** calendar years positive (1 contract, gross)"],
     charts.fig_windows(t3), charts.fig_yearly(t3), extra=f"{t3.windows_positive}/{t3.windows_total}")
src = ("MultiWalk reference table" if t7.cost_source == "multiwalk" else "the report's own commission + slippage (symbol not in our table)")
card("t7", result.verdicts["t7"],
     [f"1x cost **${t7.cost_rt_usd:.2f}** round trip per contract from {src}"
      + (f"; your report assumed ${t7.ts_cost_rt_usd:.2f}" if t7.ts_cost_rt_usd is not None and t7.cost_source == "multiwalk" else ""),
      f"gross **${t7.gross_mean_usd:,.2f}** per trade vs drift ${t7.baseline_mean_usd:,.2f} → net lift after 1x cost **${t7.net_lift_usd[1]:,.2f}** (gate > 0); net profit factor {t7.net_pf_1x:.2f}",
      f"break-even at **{t7.breakeven_mult:.1f}x** the assumed cost" if t7.breakeven_mult != float("inf") else "break-even: no cost assumed"],
     charts.fig_cost_curve(t7))
dd_lines = [f"{dd.n_trades} trades over {dd.years:.1f} years → **${dd.sum_usd:,.0f}** total, ${dd.annual_usd:,.0f}/yr (1 contract, gross)",
            f"{dd.n_episodes} drawdown episodes ({dd.n_unrecovered} unrecovered): median ${dd.median_dd:,.0f}, mean ${dd.mean_dd:,.0f}, "
            f"**CDaR-80 ${dd.cdar80:,.0f}**, max ${dd.max_dd:,.0f}",
            f"capital = 5 x CDaR-80 = **${dd.capital:,.0f}** → annual return **{dd.annual_pct:.1%}**; max drawdown = {dd.max_dd_pct_of_capital:.0%} of capital",
            f"Sharpe {dd.sharpe:.2f} · Sortino {dd.sortino:.2f} · Calmar {dd.calmar:.2f} · profit / avg DD {dd.profit_over_avg_dd:.2f} (ranking metrics)"]
if dd.worst["peak_time"] is not None:
    rec = f"{dd.worst['recovery_time']:%Y-%m-%d}" if dd.worst["recovery_time"] is not None else "not recovered"
    dd_lines.append(f"worst episode ${-dd.worst['depth']:,.0f}: peak {dd.worst['peak_time']:%Y-%m-%d} → trough {dd.worst['trough_time']:%Y-%m-%d} → {rec}")
if result.margin:
    mg = result.margin
    dd_lines.append(f"if deployed today at ${mg['today_margin_usd']:,.0f} margin: margin-to-equity **{mg['margin_to_equity']:.1%}**, "
                    f"capital covers {mg['contracts_covered']:.1f} contracts' margin")
card("drawdown", result.verdicts["drawdown"], dd_lines, charts.fig_equity(dd), charts.fig_episode_hist(dd))

st.download_button("Download results (JSON)", data=to_json(result), file_name="robustness_results.json", mime="application/json")
