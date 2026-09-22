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
    for k in ("report", "bars", "n_perm", "seed", "margin", "capital", "multiwalk", "mw_files"):
        assert k in HELP and len(HELP[k]) > 40
    assert "iUseLegacyOptimizationTextFileFormat: true" in HELP["multiwalk"] and "Import Project Setup" in HELP["multiwalk"]
    assert "CDaR-80" in CONCEPT and "5 × CDaR-80" in CONCEPT
