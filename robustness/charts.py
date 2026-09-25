"""Plotly figures for the cards and the timing-sensitivity section. No Streamlit here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from robustness.cost_stress import CostStressResult
from robustness.drawdown import DrawdownResult
from robustness.null_entry import RandomEntryResult
from robustness.plateau_grid import PlateauWindow
from robustness.selection import SelectionWindow
from robustness.temporal import TemporalResult
from robustness.timing import DelayCurve
from robustness.wfc_grid import Band, WFCWindow

ACCENT, MUTED, GREEN, RED, PALE = "#1f5fbf", "#9a9a94", "#2e8b57", "#c0392b", "#dfe7f5"
ORANGE = "#e08a1e"
_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=20, b=40), height=320, showlegend=False)


def fig_baseline(r: RandomEntryResult) -> go.Figure:
    """Bar chart: strategy mean return vs the matched random-entry baseline, % of price."""
    fig = go.Figure(go.Bar(x=["strategy", "random entry, same holds"], y=[r.observed_mean * 100, r.baseline_mean * 100],
                           marker_color=[ACCENT, MUTED], text=[f"{r.observed_mean*100:+.3f}%", f"{r.baseline_mean*100:+.3f}%"],
                           textposition="outside"))
    fig.update_layout(**_LAYOUT, yaxis_title="mean return per trade, % of price (1 contract)")
    return fig


def fig_null_hist(r: RandomEntryResult) -> go.Figure:
    """Histogram of the T8a random-entry null distribution with the observed mean marked."""
    fig = go.Figure(go.Histogram(x=r.null * 100, nbinsx=40, marker_color=MUTED, name="random-entry sets"))
    fig.add_vline(x=r.observed_mean * 100, line_color=ACCENT, line_width=3,
                  annotation_text=f"strategy {r.observed_mean*100:+.3f}%", annotation_position="top")
    fig.update_layout(**_LAYOUT, xaxis_title=f"mean return of {r.n_trades} random entries with the strategy's holds, %",
                      yaxis_title=f"count of {r.n_perm} random sets")
    return fig


def fig_windows(t3: TemporalResult) -> go.Figure:
    """Bar chart of T3's four equal-bar-count windows, lift vs drift per window."""
    xs = [f"{w.start:%Y-%m}→{w.end:%Y-%m}" for w in t3.windows]
    lifts = [(w.lift or 0.0) * 100 for w in t3.windows]
    cols = [GREEN if (w.counted and w.lift is not None and w.lift > 0) else (RED if w.counted else MUTED) for w in t3.windows]
    fig = go.Figure(go.Bar(x=xs, y=lifts, marker_color=cols, text=[f"n={w.n}" for w in t3.windows], textposition="outside"))
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(**_LAYOUT, yaxis_title="lift vs drift, % per trade", xaxis_title="window (equal bar counts)")
    return fig


def fig_yearly(t3: TemporalResult) -> go.Figure:
    """Bar chart of $ P&L per calendar year of exit, 1 contract, gross."""
    y = t3.yearly
    fig = go.Figure(go.Bar(x=y.year.astype(str), y=y.usd, marker_color=[GREEN if v > 0 else RED for v in y.usd],
                           text=[f"n={n}" for n in y.n], textposition="outside"))
    fig.update_layout(**_LAYOUT, yaxis_title="$ per year, 1 contract, gross", xaxis_title="calendar year of exit")
    return fig


def fig_cost_curve(t7: CostStressResult) -> go.Figure:
    """Net lift vs drift as a function of the cost multiplier, with the break-even marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t7.mults, y=t7.net_lift_usd, mode="lines+markers", line_color=ACCENT, fill="tozeroy", fillcolor=PALE))
    fig.add_hline(y=0, line_color=RED, line_dash="dash")
    if np.isfinite(t7.breakeven_mult) and t7.breakeven_mult > 0:
        fig.add_vline(x=t7.breakeven_mult, line_color=MUTED, line_dash="dot", annotation_text=f"break-even {t7.breakeven_mult:.1f}x")
    fig.update_layout(**_LAYOUT, xaxis_title=f"cost multiplier (1x = ${t7.cost_rt_usd:.2f} round trip per contract)",
                      yaxis_title="net lift vs drift, $ per trade (1 contract)")
    return fig


def fig_equity(dd: DrawdownResult) -> go.Figure:
    """Cumulative $ P&L equity curve, 1 contract, with drawdown episodes shaded."""
    fig = go.Figure(go.Scatter(x=dd.times, y=dd.equity, mode="lines", line_color=ACCENT, name="equity"))
    for e in dd.episodes:
        end_i = e["recovery_i"] if e["recovery_i"] is not None else len(dd.times) - 1
        x0 = dd.times[e["peak_i"]] if e["peak_i"] >= 0 else dd.times[0]
        fig.add_vrect(x0=x0, x1=dd.times[end_i], fillcolor=RED, opacity=0.08, line_width=0)
    fig.update_layout(**_LAYOUT, yaxis_title="cumulative $ P&L, 1 contract, gross", xaxis_title="trade close")
    return fig


def fig_episode_hist(dd: DrawdownResult) -> go.Figure:
    """Histogram of drawdown episode depths with CDaR-80 and the max drawdown marked."""
    fig = go.Figure(go.Histogram(x=dd.depths, nbinsx=30, marker_color=MUTED))
    fig.add_vline(x=dd.cdar80, line_color=ACCENT, line_width=3, annotation_text=f"CDaR-80 ${dd.cdar80:,.0f}", annotation_position="top")
    fig.add_vline(x=dd.max_dd, line_color=RED, line_dash="dash", annotation_text=f"max ${dd.max_dd:,.0f}", annotation_position="bottom right")
    fig.update_layout(**_LAYOUT, xaxis_title="drawdown episode depth, $", yaxis_title="episodes")
    return fig


def fig_wfc_scatter(w: WFCWindow, metric_label: str, pick_label: str = "MultiWalk's pick", top: list[int] | None = None) -> go.Figure:
    """In-sample vs out-of-sample metric per parameter combination, the window's pick highlighted
    (named pick_label), the top in-sample combinations (iteration indices) outlined red, zero lines."""
    m = np.isfinite(w.x) & np.isfinite(w.y)
    fig = go.Figure(go.Scatter(x=w.x[m], y=w.y[m], mode="markers", marker=dict(color=MUTED, size=7, opacity=0.75), name="combinations"))
    if top:
        fig.add_trace(go.Scatter(x=w.x[top], y=w.y[top], mode="markers", name=f"top {len(top)} in-sample",
                                 marker=dict(size=13, color="rgba(0,0,0,0)", line=dict(color=RED, width=2))))
    if np.isfinite(w.x[w.pick_index]) and np.isfinite(w.y[w.pick_index]):
        fig.add_trace(go.Scatter(x=[w.x[w.pick_index]], y=[w.y[w.pick_index]], mode="markers",
                                 marker=dict(color=RED, size=13, symbol="diamond"), name=pick_label))
    fig.add_hline(y=0, line_color=ACCENT, line_width=1); fig.add_vline(x=0, line_color=ACCENT, line_width=1)
    fig.update_layout(**_LAYOUT, xaxis_title=f"in-sample {metric_label}", yaxis_title=f"out-of-sample {metric_label}")
    return fig


def _pct_rank(v: np.ndarray) -> np.ndarray:
    """Percentile rank within v (0 = worst, 100 = best; average ranks for ties); v finite."""
    return (pd.Series(v).rank().to_numpy() - 1.0) / max(1, v.size - 1) * 100.0


def fig_wfc_profile(w: WFCWindow, metric_label: str, top: list[int], labels: list[str] | None = None) -> go.Figure:
    """Ranked profile, in and out of sample on one chart: the combinations finite on both sides,
    sorted by in-sample result (best on the left). Both sides are percentile ranks within the
    window (100 = best), so windows of different lengths and ratio metrics compare: the in-sample
    rank is the falling blue line, the out-of-sample rank of the same combination an orange dot
    (hollow when it lost money out-of-sample) with a rolling mean - if the orange follows the blue
    down, the ranking held. The top in-sample combinations (iteration indices) are outlined red;
    hover shows the real values and, with `labels` (one per iteration), the parameters."""
    idx = np.flatnonzero(np.isfinite(w.x) & np.isfinite(w.y))
    fig = go.Figure()
    if idx.size == 0:
        fig.update_layout(**_LAYOUT)
        return fig
    order = idx[np.argsort(-w.x[idx], kind="stable")]
    xs = np.arange(1, order.size + 1)
    is_pct, oos_pct = _pct_rank(w.x[order]), _pct_rank(w.y[order])
    name = [labels[i] if labels else f"combination {i + 1}" for i in order]
    hover = [f"{nm}<br>in-sample {w.x[i]:,.2f} (rank {r:.0f}%)<br>out-of-sample {w.y[i]:,.2f} (rank {o:.0f}%)"
             for nm, i, r, o in zip(name, order, is_pct, oos_pct)]
    fig.add_trace(go.Scatter(x=xs, y=is_pct, mode="lines", line=dict(color=ACCENT, width=2), name="in-sample rank",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=oos_pct, mode="markers", name="out-of-sample rank", hovertext=hover, hoverinfo="text",
                             marker=dict(color=ORANGE, size=8, symbol=["circle" if w.y[i] > 0 else "circle-open" for i in order],
                                         line=dict(color=ORANGE, width=1.5))))
    fig.add_trace(go.Scatter(x=xs, y=pd.Series(oos_pct).rolling(max(3, order.size // 10), center=True, min_periods=1).mean(),
                             mode="lines", line=dict(color=ORANGE, width=2, dash="dash"), name="out-of-sample, rolling mean",
                             hoverinfo="skip"))
    pos = {i: p for p, i in enumerate(order)}
    tp = [pos[i] for i in top if i in pos]
    fig.add_trace(go.Scatter(x=xs[tp], y=oos_pct[tp], mode="markers", name=f"top {len(tp)} in-sample", hoverinfo="skip",
                             marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=RED, width=2))))
    fig.update_layout(**_LAYOUT, xaxis_title=f"combinations sorted by in-sample {metric_label}, best first",
                      yaxis_title="rank within the window, % (100 = best)", yaxis_range=[-4, 104])
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.18, x=0), margin=dict(l=40, r=20, t=50, b=40))
    return fig


def fig_wfc_bands(bands: list[Band], metric_label: str) -> go.Figure:
    """Mean out-of-sample metric per in-sample band (1 = best tenth) with +/- 1 standard error and
    the share of profitable combinations as text; a falling staircase means the ranking held."""
    fig = go.Figure(go.Bar(x=[str(b.band) for b in bands], y=[b.oos_mean for b in bands],
                           error_y=dict(type="data", array=[b.oos_se if np.isfinite(b.oos_se) else 0.0 for b in bands]),
                           marker_color=[GREEN if b.oos_mean > 0 else RED for b in bands],
                           text=[f"{b.oos_pos_share:.0%} >0" for b in bands], textposition="outside"))
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(**_LAYOUT, xaxis_title="in-sample band (1 = best tenth)", yaxis_title=f"mean out-of-sample {metric_label}")
    return fig


def fig_wfc_null(w: WFCWindow) -> go.Figure:
    """Histogram of the shifted-surface null of Spearman rho with the observed value marked."""
    fig = go.Figure(go.Histogram(x=w.null, nbinsx=30, marker_color=MUTED))
    fig.add_vline(x=w.spearman, line_color=ACCENT, line_width=3, annotation_text=f"observed ρ = {w.spearman:.2f}", annotation_position="top")
    fig.update_layout(**_LAYOUT, xaxis_title="Spearman ρ of shifted OOS surfaces", yaxis_title=f"count of {w.null.size} shifts")
    return fig


def fig_plateau(p: PlateauWindow, names: list[str], pick_params: list[float]) -> go.Figure:
    """Out-of-sample metric of the pick (first, accent) and its grid neighbours (muted), sorted."""
    labels = ["pick " + "/".join(f"{v:g}" for v in pick_params)] + [f"nb {j}" for j in range(1, len(p.oos_values))]
    order = [0] + sorted(range(1, len(p.oos_values)), key=lambda j: -np.nan_to_num(p.oos_values[j], nan=-np.inf))
    fig = go.Figure(go.Bar(x=[labels[j] for j in order], y=[p.oos_values[j] for j in order],
                           marker_color=[ACCENT if j == 0 else MUTED for j in order]))
    fig.add_hline(y=0, line_color=RED, line_width=1)
    fig.update_layout(**_LAYOUT, yaxis_title="out-of-sample metric", xaxis_title=f"pick and its {p.n_neighbours} neighbours on {'/'.join(names)}")
    return fig


def fig_selection(s: SelectionWindow) -> go.Figure:
    """Histogram of the best-of-grid null (mean daily $) with the observed best and the pick marked."""
    fig = go.Figure(go.Histogram(x=s.null_max, nbinsx=30, marker_color=MUTED))
    if np.isfinite(s.best_mean):
        fig.add_vline(x=s.best_mean, line_color=ACCENT, line_width=3, annotation_text=f"best {s.best_mean:,.1f}", annotation_position="top")
    if np.isfinite(s.pick_mean):
        fig.add_vline(x=s.pick_mean, line_color=RED, line_width=2, annotation_text=f"pick {s.pick_mean:,.1f}", annotation_position="bottom")
    fig.update_layout(**_LAYOUT, xaxis_title="best mean daily $ across the grid under no edge", yaxis_title=f"count of {s.null_max.size} resamples")
    return fig


def fig_delay_curve(curve: DelayCurve, point_value_label: str, early: DelayCurve | None = None) -> go.Figure:
    """Gross $ over alive trades vs shift in bars (k = 0 = as reported, in ACCENT), with the mean %
    return per trade on a right-hand axis; hover shows how many trades are still alive. With
    `early` (the same leg moved earlier) the x axis runs -max_k..+max_k and the negative,
    hindsight side is shaded and drawn with open markers. A shift with no alive trade is a gap
    in both traces, not a $0 point."""
    pts = ([(-p.k, p, True) for p in reversed(early.points[1:])] if early is not None else []) + \
          [(p.k, p, False) for p in curve.points]
    ks = [k for k, _, _ in pts]
    # a shift with no alive trade has no return: leave a gap, never a $0 point on the zero line
    usd = [p.total_usd if p.n_alive else None for _, p, _ in pts]
    pct = [p.mean_pct * 100 if p.n_alive else None for _, p, _ in pts]
    hover = [(f"{-k} bar(s) earlier (hindsight)" if e else f"k = {k}") +
             (f": ${p.total_usd:,.0f} gross, mean {p.mean_pct*100:+.3f}% per trade, {p.n_alive} alive / {p.n_skipped} skipped"
              if p.n_alive else f": no trade alive ({p.n_skipped} skipped)")
             for k, p, e in pts]
    colors = [ACCENT if k == 0 else MUTED for k in ks]
    symbols = ["circle-open" if e else "circle" for _, _, e in pts]
    sizes = [12 if k == 0 else 8 for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=usd, mode="lines+markers", line_color=MUTED, hovertext=hover, hoverinfo="text",
                             marker=dict(size=sizes, color=colors, symbol=symbols, line=dict(width=2, color=colors))))
    fig.add_trace(go.Scatter(x=ks, y=pct, mode="lines", line=dict(color=ACCENT, dash="dot", width=1.5), opacity=0.6,
                             yaxis="y2", hoverinfo="skip"))
    fig.add_hline(y=0, line_color=RED, line_dash="dash")
    if early is not None:
        fig.add_vrect(x0=min(ks) - 0.5, x1=-0.5, fillcolor=MUTED, opacity=0.08, line_width=0,
                      annotation_text="earlier = hindsight", annotation_position="top left")
    title = (f"{curve.kind} shift, bars (negative = earlier, positive = later)" if early is not None
             else f"{curve.kind} delay, bars (k = 0 is the report)")
    fig.update_layout(**_LAYOUT, xaxis=dict(title=title, dtick=1 if len(ks) <= 21 else 2),
                      yaxis_title=f"total $, {point_value_label}",
                      yaxis2=dict(title="mean return per trade, % (dotted)", overlaying="y", side="right", showgrid=False))
    fig.update_layout(margin=dict(l=40, r=60, t=20, b=40))
    return fig
