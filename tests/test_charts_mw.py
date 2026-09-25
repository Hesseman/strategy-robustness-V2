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