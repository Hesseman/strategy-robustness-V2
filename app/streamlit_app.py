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

from app import timing_section  # noqa: E402
from app.copy import CARDS, CONCEPT, HELP  # noqa: E402
from robustness import charts  # noqa: E402
from robustness.bars_loader import BarsFormatError, load_bars  # noqa: E402
from robustness.battery import ValidationFailed, run_battery, to_json  # noqa: E402
from robustness.multiwalk_battery import MultiWalkValidationFailed, mw_to_json, run_multiwalk_battery  # noqa: E402
from robustness.multiwalk_text import MultiWalkFormatError, parse_multiwalk_text  # noqa: E402
from robustness.report_parser import ReportFormatError, parse_report  # noqa: E402
from robustness.walkforward_db import WalkforwardDBError, parse_walkforward_db  # noqa: E402
import glob  # noqa: E402

st.set_page_config(page_title="Strategy Robustness V2", layout="wide")

PILL = {"pass": ("#2e8b57", "✓ PASS"), "fail": ("#c0392b", "✗ FAIL"), "score": ("#1f5fbf", "SCORE"),
        "reference": ("#9a9a94", "REFERENCE"), "insufficient": ("#9a9a94", "n < 30 - gate not applied"),
        "insufficient_wfc": ("#9a9a94", "too few usable points - gate not applied")}


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
def _battery(report_bytes: bytes, bars_bytes: bytes, n_perm: int, seed: int, margin: float | None, capital_usd: float | None):
    return run_battery(_parse(report_bytes), _bars(bars_bytes), n_perm=n_perm, seed=seed, today_margin_usd=margin, capital_usd=capital_usd)


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


def _md(s: str) -> str:
    """Escape $ for st.markdown / tooltips (Streamlit renders $...$ as LaTeX)."""
    return s.replace("$", chr(92) + "$")


@st.cache_data(show_spinner="Reading the MultiWalk text file...")
def _mw_grid(txt_bytes: bytes):
    return parse_multiwalk_text(txt_bytes.decode("utf-8-sig", errors="replace"))


@st.cache_data(show_spinner=False)
def _mw_groups(db_bytes: bytes):
    return parse_walkforward_db(db_bytes)


@st.cache_data(show_spinner="Running the surface tests...")
def _mw_battery(txt_bytes: bytes, db_bytes: bytes, group_no: int, n_boot: int, seed: int):
    return run_multiwalk_battery(_mw_grid(txt_bytes), _mw_groups(db_bytes), group_no=group_no, n_boot=n_boot, seed=seed)


@st.cache_data(show_spinner=False)
def demo_multiwalk() -> tuple[bytes, bytes]:
    """Synthetic 5x8x6 persistent surface + its walk-forward DB for the demo button."""
    from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
    text, sched = make_multiwalk({"BarToHold": [18, 19, 20, 21, 22], "TriggerProfitAmt": [400, 600, 800, 1000, 1200, 1400, 1600, 1800],
                                  "LockInPct": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, n_days=1500, seed=42, structure="persistent", split=1000)
    return text.encode("utf-8"), make_walkforward_db(sched)


def _mw_sample_files() -> tuple[str, str] | None:
    d = os.environ.get("SR_SAMPLE_MW_DIR")
    if not d:
        return None
    txt = glob.glob(os.path.join(d, "Optimization Files", "*_MultiWalk.txt"))
    db = os.path.join(d, "Walkforward Files", "WalkforwardData.db")
    return (txt[0], db) if txt and os.path.exists(db) else None


def _p3(p: float) -> str:
    return "n/a" if not math.isfinite(p) else (f"p = {p:.4f}" if p < 0.001 else f"p = {p:.3f}")


def _rank(pct: float, n: int) -> str:
    """Percentile rank (share of combinations below) -> '<place> of <n>', best first. 'n/a' when
    the percentile is not finite or there are no points."""
    if not math.isfinite(pct) or n == 0:
        return "n/a"
    return f"{n - int(round(pct / 100 * n))} of {n}"


def _main_section(report_bytes, bars_bytes, n_perm, seed, margin, capital_usd) -> bool:
    """Accepts the uploaded bytes and the sidebar settings; renders the five cards. Returns True
    when the battery ran, False when an error was shown instead."""
    try:
        result = _battery(report_bytes, bars_bytes, int(n_perm), int(seed), margin, capital_usd)
    except ReportFormatError as e:
        st.error(f"Report not readable: {e}"); return False
    except BarsFormatError as e:
        st.error(f"Bar file not readable: {e}"); return False
    except ValidationFailed as e:
        st.error("The report and the bars do not join - fix the inputs before any test can run.")
        st.table([{"check": c.name, "ok": "✓" if c.passed else "✗", "detail": c.detail} for c in e.checks]); return False
    except Exception as e:
        st.error("Could not process these files - check that the report is the CSV Strategy Performance Report and the bars "
                 f"are a Data Window export of the same symbol and interval. Details: {type(e).__name__}: {e}"); return False

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
    if dd.capital_basis == "fixed":
        cap_line = (f"capital = **${dd.capital:,.0f}** (your starting capital) → annual return **{dd.annual_pct:.1%}**; max drawdown = "
                    f"{dd.max_dd_pct_of_capital:.0%} of capital; 5 x CDaR-80 would be ${dd.capital_5x_cdar80:,.0f}")
    else:
        cap_line = f"capital = 5 x CDaR-80 = **${dd.capital:,.0f}** → annual return **{dd.annual_pct:.1%}**; max drawdown = {dd.max_dd_pct_of_capital:.0%} of capital"
    dd_lines = [f"{dd.n_trades} trades over {dd.years:.1f} years → **${dd.sum_usd:,.0f}** total, ${dd.annual_usd:,.0f}/yr (1 contract, gross)",
                f"{dd.n_episodes} drawdown episodes ({dd.n_unrecovered} unrecovered): median ${dd.median_dd:,.0f}, mean ${dd.mean_dd:,.0f}, "
                f"**CDaR-80 ${dd.cdar80:,.0f}**, max ${dd.max_dd:,.0f}",
                cap_line,
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
    return True


st.title("Strategy Robustness V2")
st.write("Upload a TradeStation **Strategy Performance Report** (saved as CSV) and the **bar data** it ran on "
         "(Data Window export, same symbol and interval). Nothing is stored.")

with st.expander("How we judge a strategy - the concept in one minute"):
    st.markdown(_md(CONCEPT))

with st.sidebar:
    st.header("Inputs")
    up_report = st.file_uploader("Performance report (.csv)", type=["csv", "txt"], help=_md(HELP["report"]))
    with st.popover("How to export the report"):
        st.markdown(_md(HELP["report"]))
    up_bars = st.file_uploader("Bar data (.txt / .csv)", type=["txt", "csv"], help=_md(HELP["bars"]))
    with st.popover("How to export the bars"):
        st.markdown(_md(HELP["bars"]))
    n_perm = st.select_slider("Random-entry sets (T8a)", options=[500, 1000, 2000, 5000], value=1000, help=HELP["n_perm"])
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1, help=HELP["seed"])
    margin_in = st.number_input("Today's margin per contract, $ (optional)", min_value=0.0, value=0.0, step=50.0, help=HELP["margin"])
    capital_basis = st.radio("Capital for the drawdown card", ["5 × CDaR-80 (recommended)", "Fixed starting capital"],
                             key="capital_basis", help=_md(HELP["capital"]))
    capital_usd = None
    if capital_basis == "Fixed starting capital":
        capital_usd = float(st.number_input("Starting capital, $", min_value=100.0, value=10_000.0, step=500.0, key="capital_usd_in",
                                            help=_md(HELP["capital_usd"])))
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

main_ready = bool(report_bytes and bars_bytes)
if not main_ready:
    st.info("Upload both files to start the main battery, or click **Try the demo** in the sidebar.")

main_ok = False
if main_ready:
    main_ok = _main_section(report_bytes, bars_bytes, n_perm, seed, float(margin_in) or None, capital_usd)
if main_ok:
    st.divider()
    timing_section.render(report_bytes, bars_bytes)

st.divider()
st.header("MultiWalk surface tests (optional)")
st.caption("Three more cards for strategies optimised in MultiWalk Pro: they read every parameter combination, not just the winner. "
           "No report or bars needed.")
mw_sample = _mw_sample_files()
with st.expander("What this needs and how to get the two files", expanded=True):
    st.markdown(_md(HELP["multiwalk"]))
c1, c2 = st.columns(2)
up_txt = c1.file_uploader("Optimisation text file (…_MultiWalk.txt)", type=["txt"], help=_md(HELP["mw_files"]), key="mw_txt")
up_db = c2.file_uploader("Walk-forward database (WalkforwardData.db)", type=["db"], help=_md(HELP["mw_files"]), key="mw_db")
b1, b2, b3, b4 = st.columns(4)
n_boot = b1.select_slider("Resamples (selection haircut)", options=[200, 500, 1000], value=500, help=HELP["n_boot"])
if mw_sample and b2.button("Load MultiWalk sample (dev)"):
    st.session_state["mw_sample"] = True
if b3.button("Try the MultiWalk demo (synthetic)"):
    st.session_state["mw_demo"] = True
if (st.session_state.get("mw_demo") or st.session_state.get("mw_sample")) and b4.button("Clear MultiWalk data"):
    st.session_state.pop("mw_demo", None); st.session_state.pop("mw_sample", None); st.rerun()
if st.session_state.get("mw_demo"):
    txt_bytes, db_bytes = demo_multiwalk()
    st.caption("synthetic demo - a planted smooth surface that persists out-of-sample, not a real strategy")
elif st.session_state.get("mw_sample") and mw_sample:
    txt_bytes, db_bytes = Path(mw_sample[0]).read_bytes(), Path(mw_sample[1]).read_bytes()
    st.caption(f"sample: {Path(mw_sample[0]).name} + {Path(mw_sample[1]).name}")
else:
    txt_bytes = up_txt.getvalue() if up_txt else None
    db_bytes = up_db.getvalue() if up_db else None

if not (txt_bytes and db_bytes):
    st.info("Upload both MultiWalk files to run the surface tests.")
    st.stop()

try:
    groups = _mw_groups(db_bytes)
    group_no = groups[0].group_no
    if len(groups) > 1:
        labels = {g.label: g.group_no for g in groups}
        group_no = labels[st.selectbox("Walk-forward group", list(labels))]
    mw = _mw_battery(txt_bytes, db_bytes, int(group_no), int(n_boot), int(seed))
except MultiWalkFormatError as e:
    st.error(f"Text file not readable: {e}"); st.stop()
except WalkforwardDBError as e:
    st.error(f"Database not readable: {e}"); st.stop()
except MultiWalkValidationFailed as e:
    st.error("The text file and the database do not describe the same optimisation.")
    st.table([{"check": c.name, "ok": "✓" if c.passed else "✗", "detail": c.detail} for c in e.checks]); st.stop()
except Exception as e:
    st.error(f"Could not process the MultiWalk files. Details: {type(e).__name__}: {e}"); st.stop()

mm = mw.meta
st.subheader(f"{mm['strategy'] or 'MultiWalk project'} · {mm['symbol']} {mm['interval']} · {mm['n_iter']} combinations on "
             f"{' × '.join(mm['param_names'])} ({' × '.join(str(s) for s in mm['shape'])}) · {mm['dates_start']:%Y-%m-%d} → {mm['dates_end']:%Y-%m-%d}")
st.caption(f"{mm['group']} · metric: {mm['fitness']} ({mm['metric']}) · {mm['n_complete']} of {mm['n_windows']} windows complete · "
           f"{mm['n_null']} null shifts, {mm['n_boot']} resamples, seed {mm['seed']}")
st.markdown(f"**Gates passed: {mw.gates_passed} of {mw.gates_total}.** " + mw.caveat)
with st.expander("Validation checks", expanded=not all(c.passed for c in mw.checks)):
    st.table([{"check": c.name, "ok": "✓" if c.passed else ("⚠" if c.severity == "warn" else "✗"), "detail": c.detail} for c in mw.checks])
rows = []
for w, pw, sw in zip(mw.wfc.windows, mw.plateau.windows, mw.selection.windows):
    rows.append({"window": w.label, "complete": "✓" if w.complete else "✗", "points": w.n_points, "Spearman ρ": round(w.spearman, 3),
                 "Pearson r": round(w.pearson, 3), "p": w.p_value if w.null.size else None, "positive OOS share": round(w.pos_oos_frac, 2),
                 "quadrant": w.quadrant, "plateau (OOS)": None if not math.isfinite(pw.score_oos) else round(pw.score_oos, 2), "pick deflated p": None if not math.isfinite(sw.p_pick) else sw.p_pick,
                 "pick IS rank %": None if not math.isfinite(w.pick_is_pct) else round(w.pick_is_pct), "pick OOS rank %": None if not math.isfinite(w.pick_oos_pct) else round(w.pick_oos_pct)})
st.dataframe(rows, width="stretch", hide_index=True)
usable = [w for w in mw.wfc.windows if w.complete and not w.insufficient]
chart_labels = [w.label for w in mw.wfc.windows]
default_i = mw.wfc.windows.index(usable[-1]) if usable else 0
sel = st.selectbox("Window to chart", chart_labels, index=default_i)
wi = chart_labels.index(sel)
ww, pw, sw = mw.wfc.windows[wi], mw.plateau.windows[wi], mw.selection.windows[wi]
metric_label = "NP / avg DD" if mm["metric"] == "NPAvgDD" else "net profit $"

wfc_verdict = mw.verdicts["wfc"] if mw.verdicts["wfc"] != "insufficient" else "insufficient_wfc"
wfc_lines = [(f"pooled over **{len(usable)} complete window(s)**: Spearman ρ = **{mw.wfc.pooled_spearman:.2f}** → {_p3(mw.wfc.pooled_p)} (gate < 0.05); "
              f"positive OOS among positive-IS combinations **{mw.wfc.pooled_pos_oos_frac:.0%}** (gate ≥ 50%)") if usable else "no complete window with enough usable points"]
for w in mw.wfc.windows:
    wfc_lines.append(f"{w.label}: ρ = {w.spearman:.2f} (Pearson {w.pearson:.2f}), {_p3(w.p_value) if w.null.size else 'too few points'}, "
                     f"{w.n_points} points → *{w.quadrant}*; MultiWalk's pick: IS rank {_rank(w.pick_is_pct, w.n_points)}, OOS rank {_rank(w.pick_oos_pct, w.n_points)}"
                     if math.isfinite(w.pick_is_pct) else f"{w.label}: ρ = {w.spearman:.2f}, {w.n_points} points")
if mw.wfc_np is not None and usable:
    wfc_lines.append(f"same test on net profit: pooled ρ = {mw.wfc_np.pooled_spearman:.2f}, {_p3(mw.wfc_np.pooled_p)}")
card("wfc", wfc_verdict, wfc_lines, charts.fig_wfc_scatter(ww, metric_label), charts.fig_wfc_null(ww) if ww.null.size else None,
     extra=f"ρ = {mw.wfc.pooled_spearman:.2f}" if usable else "")

pl_lines = [f"pooled plateau score **{mw.plateau.pooled_score:.2f}** over {mw.plateau.n_complete} complete window(s) (1 = flat and all neighbours profitable, 0 = spike)"
            if math.isfinite(mw.plateau.pooled_score) else "pooled plateau score n/a - no complete window has enough usable points"]
for p in mw.plateau.windows:
    if not math.isfinite(p.score_oos):
        pl_lines.append(f"{p.label}: n/a - fewer than 2 usable points among the pick and its {p.n_neighbours} neighbours"); continue
    pl_lines.append(f"{p.label}: OOS score {p.score_oos:.2f} (flatness {p.flat_oos:.2f}, {p.positive_share_oos:.0%} of {p.n_neighbours} neighbours profitable); IS score {p.score_is:.2f}")
card("plateau", "score", pl_lines, charts.fig_plateau(pw, mm["param_names"], mm["windows"][wi]["params"]),
     extra=f"{mw.plateau.pooled_score:.2f}" if math.isfinite(mw.plateau.pooled_score) else "")

se_lines = []
for s in mw.selection.windows:
    if s.insufficient:
        se_lines.append(f"{s.label}: not run (incomplete window or too few in-sample days)"); continue
    se_lines.append(f"{s.label}: best of {s.n_iter} in-sample = combination {s.best_index + 1} at **${s.best_mean:,.1f}/day**, deflated {_p3(s.p_best)}; "
                    f"MultiWalk's pick ${s.pick_mean:,.1f}/day, deflated {_p3(s.p_pick)}; effective independent variants ≈ **{s.n_eff:.1f}** of {s.n_iter}")
card("selection", "reference", se_lines, charts.fig_selection(sw))
st.download_button("Download MultiWalk results (JSON)", data=mw_to_json(mw), file_name="multiwalk_surface_results.json", mime="application/json")
