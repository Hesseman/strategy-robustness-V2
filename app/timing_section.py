"""Timing sensitivity - one section of the main page. Reuses the report and bars already
uploaded for the five cards: delayed-entry and delayed-exit curves, plus earlier (hindsight)
curves for reference. A picture of fragility, not a gate: no verdict pill, no p-value, not
counted in 'Gates passed'.

Only reason to change: how the timing result is presented. The delay conventions live in
robustness/timing.py."""
from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from app.copy import CARDS, HELP
from app.loaders import loaded_bars, parsed_report
from robustness import charts
from robustness.timing import DelayCurve, TimingResult, paired_change, run_timing, to_json

MODE_LABELS = {"fixed_exit": "Fixed exit - a delayed entry keeps the reported exit",
               "fixed_hold": "Fixed hold - a delayed entry moves the exit too"}


def _esc(s: str) -> str:
    """Escape '$' so st.markdown does not read a pair of dollar amounts as LaTeX."""
    return s.replace("$", chr(92) + "$")


def _usd(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _dusd(v: float) -> str:
    """Signed $ change: +$1,605 / -$5,537."""
    return ("+" if v >= 0 else "") + _usd(v)


def _leg(c: DelayCurve) -> str:
    """'the entry', 'the exit' or 'the whole trade': an entry curve in fixed_hold mode moves
    both legs, so its shifts are whole-trade shifts."""
    if c.kind == "exit":
        return "the exit"
    return "the entry" if c.mode == "fixed_exit" else "the whole trade"


@st.cache_data(show_spinner="Computing the delay curves...")
def _timing(report_bytes: bytes, bars_bytes: bytes, max_k: int, mode: str) -> TimingResult:
    return run_timing(parsed_report(report_bytes), loaded_bars(bars_bytes), max_k=max_k, mode=mode)


def early_line(e: DelayCurve, c: DelayCurve) -> str:
    """Accepts the earlier (hindsight) curve and the delayed curve of the same leg and mode;
    returns one bullet: what acting 1 bar earlier does to each trade that still fits, compared
    with the same trade as reported, and to the total. Guarantees the wording names what moved
    (the entry, the exit, or the whole trade in fixed_hold) and that trades too short to shift
    are counted, never averaged in."""
    verb = {"the entry": "entering", "the exit": "exiting", "the whole trade": "shifting the whole trade"}[_leg(e)]
    p0, p1 = c.points[0], e.points[1]
    mean, n = paired_change(e, 1)
    if n == 0:
        return f"hindsight: {verb} 1 bar earlier fits no trade"
    short = p0.n_alive - n
    return (f"hindsight: {verb} 1 bar earlier changes each of the {n} trades that fit by **{_dusd(mean)}** "
            f"(total {_usd(p1.total_usd)}, {_dusd(p1.total_usd - p0.total_usd)} vs as reported"
            + (f"; {short} trades too short to shift" if short else "") + ")")


def where_lines(r: TimingResult) -> list[str]:
    """Accepts a TimingResult computed in 'fixed_exit' mode, where every curve moves one leg and
    keeps the other as reported; returns the 'where timing matters' bullets: for each leg moved
    1 bar earlier (hindsight) and 1 bar later, the change in the total and the change per trade
    on the trades that still fit, each compared with itself as reported; then which leg the
    earlier side favours, judged on those paired per-trade changes. Guarantees an empty list
    when any 1-bar shift fits no trade; raises ValueError for a 'fixed_hold' result, whose entry
    curves move both legs."""
    if r.entry.mode != "fixed_exit":
        raise ValueError("where_lines compares single legs: pass the fixed_exit result")
    p0 = r.entry.points[0]
    curves = {"ee": r.entry_early, "xe": r.exit_early, "el": r.entry, "xl": r.exit}
    paired = {key: paired_change(c, 1) for key, c in curves.items()}
    if min(n for _, n in paired.values()) == 0:
        return []

    def fmt(key: str) -> str:
        mean, n = paired[key]
        return f"{_dusd(curves[key].points[1].total_usd - p0.total_usd)} total, {_dusd(mean)} per trade on the {n} that fit"
    lines = [f"1 bar **earlier** (hindsight): entry {fmt('ee')}; exit {fmt('xe')}",
             f"1 bar **later**: entry {fmt('el')}; exit {fmt('xl')}"]
    d_e, d_x = paired["ee"][0], paired["xe"][0]
    if max(d_e, d_x) <= 0:
        lines.append("neither leg gains from acting earlier: the signals are not late relative to the move")
    else:
        leg, other = ("entry", "exit") if d_e >= d_x else ("exit", "entry")
        lines.append(f"an earlier **{leg}** gains more than an earlier {other}: if you work on one trigger, "
                     f"this curve says the {leg} lags the move more")
    return lines


def result_lines(c: DelayCurve, n_trades: int) -> list[str]:
    """Accepts a DelayCurve and the trade count; returns the card's result bullets: what the
    first delay keeps of the total (or how it changes the total, signed, when there is no
    positive baseline) and how it changes each trade that still fits, compared with itself as
    reported; where the profit turns non-positive; and how many trades are still alive at the
    largest delay. Guarantees no ratio is shown without a positive as-reported total and trades
    that no longer fit are counted, never averaged in; relies on k = 0 keeping every trade alive."""
    leg = _leg(c)
    p0, p1, pk = c.points[0], c.points[1], c.points[-1]
    lines = [f"as reported (k = 0): **{_usd(p0.total_usd)}** gross over {p0.n_alive} trades, "
             f"mean {p0.mean_pct*100:+.3f}% per trade, win rate {p0.win_rate:.0%}"]
    mean, n = paired_change(c, 1)
    fit = (f"; the {n} trades that still fit change by **{_dusd(mean)}** each" if n else "") + \
          (f", {p1.n_skipped} no longer fit" if p1.n_skipped else "")
    if p1.n_alive == 0:
        lines.append(f"delaying {leg} 1 bar leaves no trade that fits")
    elif math.isfinite(p1.retention):
        lines.append(f"delaying {leg} 1 bar keeps **{p1.retention:.0%}** of the gross $ "
                     f"({_usd(p1.total_usd)} of {_usd(p0.total_usd)}){fit}")
    else:
        lines.append(f"delaying {leg} 1 bar changes the gross $ by **{_dusd(p1.total_usd - p0.total_usd)}** "
                     f"({_usd(p0.total_usd)} → {_usd(p1.total_usd)}; no ratio - the as-reported total is not positive){fit}")
    alive_ks = [p.k for p in c.points if p.n_alive > 0]
    if p0.total_usd <= 0:
        lines.append("not profitable as reported, so there is no profit for a delay to lose")
    elif c.first_nonpositive_k is not None:
        lines.append(f"profit turns non-positive at **k = {c.first_nonpositive_k}**")
    else:
        last = alive_ks[-1]
        lines.append(f"stays positive through **k = {last}**"
                     + (f" (no trade alive beyond k = {last})" if last < pk.k else ""))
    lines.append(f"n alive at k = {pk.k}: **{pk.n_alive} of {n_trades}**")
    return lines


def timing_card(key: str, c: DelayCurve, e: DelayCurve, n_trades: int, fig: go.Figure) -> None:
    """Accepts a key from CARDS, its delayed and earlier (hindsight) DelayCurves, the trade
    count and the plotly figure; renders one bordered card (copy and result lines left, chart
    right). No verdict pill. Guarantees an entry card in fixed_hold mode says it shifts the
    whole trade, since its static copy describes fixed_exit."""
    cp = CARDS[key]
    with st.container(border=True):
        st.markdown(f"### {cp['title']}")
        st.caption(cp["tagline"])
        if _leg(c) == "the whole trade":
            st.caption("Fixed hold: the exit moves with the entry, so this card shifts the whole trade, on both sides of the chart.")
        left, right = st.columns([1, 1.2])
        with left:
            st.markdown(_esc(f"**What it shows.** {cp['catches']}"))
            st.markdown(_esc(f"**How it works.** {cp['how']}"))
            st.markdown("**Result**")
            for line in result_lines(c, n_trades) + [early_line(e, c)]:
                st.markdown(_esc(f"- {line}"))
        with right:
            st.plotly_chart(fig, width="stretch")


def _section_body(report_bytes: bytes, bars_bytes: bytes, max_k: int, mode: str) -> None:
    result = _timing(report_bytes, bars_bytes, max_k, mode)
    single = result if mode == "fixed_exit" else _timing(report_bytes, bars_bytes, max_k, "fixed_exit")
    m = result.meta
    pv_label = f"1 contract at {m['point_value']:g} USD/pt, gross"
    st.caption(_esc(f"{m['n_trades']} trades · {pv_label}"))
    st.markdown(_esc(result.caveat))
    share = result.at_open_share
    st.markdown(_esc(f"**{share:.0%} of entries were filled at the bar open**, so the entry card's 0 → 1 step is "
                     + ("a pure timing effect." if share >= 0.95 else
                        "partly a fill-price effect (report price → next bar's open), not only timing.")
                     + " Exit fills are not checked: if exits were stop or limit fills inside the bar, the exit card's "
                       "0 → 1 step also carries a fill-price effect."))
    wl = where_lines(single)
    if wl:
        st.markdown(_esc("**Where timing matters** - one leg moved, the other as reported (whichever mode is selected); "
                         "gross $, one contract. 'Per trade' compares each trade that still fits with itself as reported, "
                         "so trades too short to shift cannot move it; the total also loses them."))
        for line in wl:
            st.markdown(_esc(f"- {line}"))
    timing_card("timing_entry", result.entry, result.entry_early, m["n_trades"],
                charts.fig_delay_curve(result.entry, pv_label, early=result.entry_early))
    timing_card("timing_exit", result.exit, result.exit_early, m["n_trades"],
                charts.fig_delay_curve(result.exit, pv_label, early=result.exit_early))
    st.download_button("Download timing results (JSON)", data=lambda: to_json(result),
                       file_name="timing_results.json", mime="application/json", on_click="ignore")


def render(report_bytes: bytes, bars_bytes: bytes) -> None:
    """Accepts the report and bar bytes the five cards already ran on; draws the timing
    section: header, its two controls (largest shift, delayed-entry mode), the caveat, the
    at-open share, the 'where timing matters' lines, the entry-delay and exit-delay cards and
    a JSON download built only when clicked. Returns nothing. Guarantees it never calls
    st.stop() - the MultiWalk section below must still render - and that any failure inside
    the section shows one warning and returns; the controls keep their values while the
    section is hidden (no files, a failed join)."""
    st.header("Timing sensitivity (reference)")
    st.caption(_esc(HELP["timing"]))
    c1, c2 = st.columns([1, 2])
    # "session", not "page": the app has one page, and "page" dropped the value on re-mount
    max_k = c1.slider("Largest shift, bars", min_value=1, max_value=20, value=10, step=1,
                      key="timing_max_k", persist_state="session", help=HELP["timing_max_k"])
    mode = c2.radio("Delayed entry", options=list(MODE_LABELS), format_func=MODE_LABELS.get,
                    key="timing_mode", horizontal=True, persist_state="session", help=HELP["timing_mode"])
    try:
        _section_body(report_bytes, bars_bytes, int(max_k), str(mode))
    except Exception as e:  # the cards above already joined these files, so this is a bug path
        st.warning(_esc(f"Timing section skipped: {type(e).__name__}: {e}"))
