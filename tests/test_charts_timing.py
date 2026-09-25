"""fig_delay_curve on synthetic timing results: one point per shift, k = 0 in ACCENT, the earlier side open-marked."""
from robustness import charts
from robustness.bars_loader import load_bars
from robustness.report_parser import parse_report
from robustness.synthetic import bars_to_ts_text, make_bars, make_trades, trades_to_report_text
from robustness.timing import run_timing


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


def test_fig_delay_curve_leaves_gaps_where_no_trade_is_alive():
    bars = make_bars(n=4000, seed=30)
    trades = make_trades(bars, n=80, seed=31, max_hold=3)
    r = run_timing(parse_report(trades_to_report_text(trades)), load_bars(bars_to_ts_text(bars)), max_k=10)
    fig = charts.fig_delay_curve(r.entry, "1 contract, gross", early=r.entry_early)
    dead = [i for i, k in enumerate(fig.data[0].x) if k > 0 and r.entry.points[k].n_alive == 0]
    assert dead, "holds of at most 3 bars must leave delayed-entry shifts with no alive trade"
    assert all(fig.data[0].y[i] is None and fig.data[1].y[i] is None for i in dead)
    assert all("no trade alive" in fig.data[0].hovertext[i] for i in dead)
