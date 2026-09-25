"""A portable three-window MultiWalk run: two complete windows plus one incubation window whose
nominal OOS end lies far beyond the data. Covers what the single-window synthetic cannot - pooled
statistics over several windows, the incomplete-window paths through every test and every figure,
and NaN serialisation when nothing is usable at all."""
import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from robustness import charts
from robustness.multiwalk_battery import mw_to_json, run_multiwalk_battery
from robustness.multiwalk_text import parse_multiwalk_text
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db

_N_DAYS = 1500
_IN_LEN = 400                                   # calendar days, period type Day
_OUT_LEN = 300                                  # calendar days, period type Day
_PERIOD_DAY, _FITNESS_NPAVGDD = 7, 6
_OOS_DAYS = [(500, 799), (800, 1099), (1400, 1499)]   # trading-day indices, third one is the incubation window


def _inputs():
    """The grid plus a hand-built three-window schedule in make_multiwalk's own shape."""
    text, _ = make_multiwalk({"A": list(range(5)), "B": list(range(4))}, n_days=_N_DAYS, seed=17,
                             structure="persistent", split=1000)
    grid = parse_multiwalk_text(text)
    dates = pd.bdate_range("2020-01-06", periods=_N_DAYS)
    windows = []
    for i, (first, last) in enumerate(_OOS_DAYS):
        oos_start = dates[first]
        is_mask = np.asarray((dates >= oos_start - pd.DateOffset(days=_IN_LEN))
                             & (dates <= oos_start - pd.Timedelta(days=1)), dtype=bool)
        row = int(np.argmax(grid.daily_pnl[:, is_mask].sum(axis=1))) + 1   # best in-sample net profit
        windows.append({"oos_start": oos_start.strftime("%Y%m%d"),
                        "oos_end": "20300101" if i == 2 else dates[last].strftime("%Y%m%d"),
                        "params": tuple(grid.params[row - 1]), "grid_row": row})
    schedule = {"param_names": grid.param_names, "n_iter": grid.n_iter, "in_len": _IN_LEN, "in_type_id": _PERIOD_DAY,
                "out_len": _OUT_LEN, "out_type_id": _PERIOD_DAY, "anchored": False, "fitness_id": _FITNESS_NPAVGDD,
                "symbol": "@SYN", "interval": "30min", "strategy": "Synthetic_MW_3W", "windows": windows}
    return grid, parse_walkforward_db(make_walkforward_db(schedule))


def test_three_windows_pool_over_the_complete_two_and_still_report_the_incubation_window():
    grid, groups = _inputs()
    r = run_multiwalk_battery(grid, groups, n_null=60, n_boot=30, seed=0)

    assert len(r.windows) == 3 and r.meta["n_complete"] == 2
    assert r.windows[2].complete is False
    assert r.selection.windows[2].insufficient is True and math.isnan(r.selection.windows[2].p_pick)
    assert r.wfc.windows[2].n_points > 0            # 100 OOS days, a trade every 7 days: still usable
    assert math.isfinite(r.wfc.pooled_p) and 0 < r.wfc.pooled_p <= 1
    assert r.wfc.n_null == 60 and r.plateau.n_complete == 2
    assert r.region.n_complete == 2 and len(r.region.windows) == 3 and math.isfinite(r.region.windows[2].lift)
    assert r.region.lift == np.mean([r.region.windows[0].lift, r.region.windows[1].lift])   # the incubation window is not pooled
    assert [w["wfc_reading"] for w in r.meta["windows"]][:2] == ["edge", "edge"] and r.verdicts["wfc"] == "pass"

    figs = (charts.fig_wfc_scatter(r.wfc.windows[2], "NP"), charts.fig_wfc_null(r.wfc.windows[2]),
            charts.fig_plateau(r.plateau.windows[2], r.meta["param_names"], r.meta["windows"][2]["params"]),
            charts.fig_selection(r.selection.windows[2]))
    assert all(isinstance(f, go.Figure) for f in figs)

    json.dumps(json.loads(mw_to_json(r)), allow_nan=False)


def test_nothing_usable_reports_nan_everywhere_and_still_serialises():
    grid, groups = _inputs()
    r = run_multiwalk_battery(grid, groups, n_null=60, n_boot=30, seed=0, min_trades=10**6)

    assert r.wfc.reasons == ["wfc_insufficient"] and math.isnan(r.wfc.pooled_p)   # the continuity correlation has nothing left
    assert r.verdicts["wfc"] != "insufficient" and r.region.n_complete == 2         # the region lift has no trade-count hole
    assert all(math.isnan(p.score_oos) for p in r.plateau.windows)
    assert math.isnan(r.plateau.pooled_score)

    json.dumps(json.loads(mw_to_json(r)), allow_nan=False)
