import plotly.graph_objects as go

from robustness import charts
from robustness.multiwalk_battery import run_multiwalk_battery
from robustness.multiwalk_text import parse_multiwalk_text
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db


def test_multiwalk_figures_build():
    text, sched = make_multiwalk({"A": list(range(4)), "B": list(range(3))}, n_days=400, seed=4, structure="persistent", split=260)
    r = run_multiwalk_battery(parse_multiwalk_text(text), parse_walkforward_db(make_walkforward_db(sched)), n_null=20, n_boot=10)
    w, p, s = r.wfc.windows[0], r.plateau.windows[0], r.selection.windows[0]
    for fig in (charts.fig_wfc_scatter(w, "NP/Avg DD"), charts.fig_wfc_null(w), charts.fig_plateau(p, ["A", "B"], r.meta["windows"][0]["params"]),
                charts.fig_selection(s)):
        assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_copy_has_the_new_cards_and_help():
    from app.copy import CARDS, CONCEPT, HELP
    assert {"wfc", "plateau", "selection"} <= set(CARDS)
    for k in ("report", "bars", "n_perm", "seed", "margin", "capital", "capital_usd", "multiwalk", "mw_files"):
        assert k in HELP and len(HELP[k]) > 40
    assert "iUseLegacyOptimizationTextFileFormat: true" in HELP["multiwalk"] and "Import Project Setup" in HELP["multiwalk"]
    assert "CDaR-80" in CONCEPT and "5 × CDaR-80" in CONCEPT


def test_ranked_profile_and_bands_figures():
    import numpy as np
    from robustness.wfc_grid import WFCWindow, top_n_summary, wfc_bands
    x = np.arange(1.0, 13.0); x[3] = np.nan
    y = np.array([3.0, -1.0, 2.0, 5.0, -2.0, 4.0, 1.0, 6.0, -3.0, 7.0, 0.0, 8.0])
    w = WFCWindow(1, "w", True, 11, 1, x, y, 0.0, 0.0, 0, 0.0, 1.0, np.empty(0), 11, float("nan"), float("nan"), "", False)
    top = top_n_summary(w).indices
    fig = charts.fig_wfc_profile(w, "net profit $", top, labels=[f"p{i}" for i in range(12)])
    is_tr, oos_tr = fig.data[0], fig.data[1]
    assert list(is_tr.x) == list(range(1, 12)) and is_tr.y[0] == 100.0 and min(is_tr.y) == 0.0
    assert all(0.0 <= v <= 100.0 for v in oos_tr.y)
    order = [i for i in np.argsort(-np.nan_to_num(x, nan=-np.inf), kind="stable") if np.isfinite(x[i])]
    assert list(oos_tr.marker.symbol) == ["circle" if y[i] > 0 else "circle-open" for i in order]
    outline = next(t for t in fig.data if t.name and t.name.startswith("top "))
    assert len(outline.x) == len(top) == 3
    bands = wfc_bands(w, 4)
    assert len(charts.fig_wfc_bands(bands, "net profit $").data[0].x) == 4
    scatter = charts.fig_wfc_scatter(w, "net profit $", pick_label="best in-sample", top=top)
    assert any(t.name == "best in-sample" for t in scatter.data) and any((t.name or "").startswith("top ") for t in scatter.data)


def test_scatter_draws_tinsleys_best_fit_line():
    import numpy as np
    from robustness.wfc_grid import WFCWindow
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([2.5, 3.9, 9.0, 8.2, 9.8, 12.1])
    w = WFCWindow(1, "w", True, 5, 1, x, y, 0.0, 0.0, 0, 0.0, 1.0, np.empty(0), 0, float("nan"), float("nan"), "", False)
    fit = next(t for t in charts.fig_wfc_scatter(w, "Sharpe").data if t.name == "best fit")
    m = np.isfinite(x)
    slope, intercept = np.polyfit(x[m], y[m], 1)
    assert list(fit.x) == [1.0, 6.0] and np.allclose(fit.y, [intercept + slope * 1.0, intercept + slope * 6.0])
    assert fit.line.color == charts.GREEN
    flat = WFCWindow(1, "w", True, 3, 0, np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0.0, 0.0, 0, 0.0, 1.0,
                     np.empty(0), 0, float("nan"), float("nan"), "", False)
    assert not any(t.name == "best fit" for t in charts.fig_wfc_scatter(flat, "Sharpe").data)   # no line through a vertical stack

def _region_window(shape, seed=1):
    from robustness.region_wfc import region_test
    from robustness.synthetic_multiwalk import make_planted_grid
    from robustness.windows import custom_windows
    grid = make_planted_grid(shape, structure="ridge", rho=0.6, trade_p=0.3, n_days=300, seed=seed)
    return grid, region_test(grid, custom_windows(grid.dates, "single"), n_null=10, seed=0).windows[0]


def test_region_surfaces_draw_four_panels_with_each_ridge_outlined():
    """5x5 grid (index = a * 5 + b; the first axis is drawn across, the second up). The pooled
    in-sample panel gets a plus-shaped peak - its top fifth is exactly the five plus cells, whose
    outline has 12 unit edges; the raw in-sample panel is flat except one cell, so its ridge
    (>= the 80th percentile, i.e. every cell) is the whole square: 20 edges."""
    from dataclasses import replace

    import numpy as np
    grid, rw = _region_window((5, 5))
    at = {tuple(p): i for i, p in enumerate(grid.grid_pos.tolist())}
    plus = [at[(2, 2)], at[(1, 2)], at[(3, 2)], at[(2, 1)], at[(2, 3)]]
    xp = np.zeros(25); xp[plus] = [9.0, 8.0, 8.0, 7.0, 7.0]
    x = np.zeros(25); x[at[(4, 0)]] = 5.0
    rw = replace(rw, x=x, x_pooled=xp, region=np.array(plus[:5]))
    fig = charts.fig_region_surfaces(rw, grid.grid_pos, grid.axes, grid.param_names)
    heat = {t.name: t for t in fig.data if t.type == "heatmap"}
    assert set(heat) == {"in-sample, raw", "out-of-sample, raw", "in-sample, pooled", "out-of-sample, pooled"}
    z = np.array(heat["in-sample, pooled"].z, dtype=float)
    assert z.shape == (5, 5) and z[2, 2] == 9.0 and z[2, 1] == 8.0 and z[1, 2] == 7.0   # row = second parameter, column = first
    outline = {t.name: t for t in fig.data if t.type == "scatter"}
    assert list(outline["ridge: in-sample, pooled"].x).count(None) == 12
    assert list(outline["ridge: in-sample, raw"].x).count(None) == 20
    assert fig.layout.coloraxis.cmin == -fig.layout.coloraxis.cmax                       # centred on zero


def test_region_surfaces_slice_a_three_parameter_grid_through_the_best_pooled_cell():
    import numpy as np
    grid, rw = _region_window((4, 3, 2), seed=2)
    fig = charts.fig_region_surfaces(rw, grid.grid_pos, grid.axes, grid.param_names)
    best = grid.grid_pos[rw.region[0]]
    assert f"p2={grid.axes[2][best[2]]:g}" in fig.layout.title.text
    z = np.array(next(t for t in fig.data if t.name == "in-sample, raw").z, dtype=float)
    assert z.shape == (3, 4)
    at = {tuple(p): i for i, p in enumerate(grid.grid_pos.tolist())}
    assert all(z[r, c] == rw.x[at[(c, r, best[2])]] for r in range(3) for c in range(4))
    one_d, rw1 = _region_window((7,), seed=3)
    z1 = np.array(next(t for t in charts.fig_region_surfaces(rw1, one_d.grid_pos, one_d.axes, one_d.param_names).data
                        if t.name == "out-of-sample, raw").z, dtype=float)
    assert z1.shape == (1, 7) and np.allclose(z1[0], rw1.y)


def test_region_null_histogram_marks_the_observed_lift():
    import numpy as np
    _, rw = _region_window((5, 5))
    fig = charts.fig_region_null(rw)
    assert np.allclose(fig.data[0].x, rw.null_lift) and fig.layout.shapes[0].x0 == rw.lift


def test_base_setting_chart_marks_the_centre_the_ensemble_and_kaufmans_cell():
    import numpy as np
    from robustness.base_setting import base_setting
    from robustness.synthetic_multiwalk import make_planted_grid
    from robustness.windows import custom_windows
    grid = make_planted_grid((7, 7), structure="persistent", rho=0.6, trade_p=0.3, n_days=400, seed=4)
    b = base_setting(grid, custom_windows(grid.dates, "two"), "edge", "two")
    fig = charts.fig_base_setting(b, grid.grid_pos, grid.axes, grid.param_names)
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 1 and np.allclose(np.array(heat[0].z, dtype=float)[grid.grid_pos[:, 1], grid.grid_pos[:, 0]], b.pooled)
    marks = {t.name: t for t in fig.data if t.type == "scatter" and t.mode == "markers"}
    c = grid.grid_pos[b.centre]
    assert (marks["base setting"].x[0], marks["base setting"].y[0]) == (c[0], c[1])
    assert len(marks["ensemble"].x) == len(b.ensemble) - 1 and "Kaufman's average (display only)" in marks
    assert list(next(t for t in fig.data if t.name == "region").x).count(None) > 0
    assert b.basis_start.strftime("%Y-%m-%d") in fig.layout.title.text
