import sqlite3
import tempfile
import os

import numpy as np

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db

AXES = {"A": [1, 2, 3], "B": [10, 20]}


def test_text_round_trips_through_the_parser():
    text, sched = make_multiwalk(AXES, n_days=120, seed=1, structure="persistent", split=80)
    g = parse_multiwalk_text(text)
    assert g.param_names == ["A", "B"] and g.n_iter == 6 and g.shape == (3, 2) and g.is_full_grid
    assert g.params[1].tolist() == [2, 10]          # first input varies fastest
    assert g.n_days == 120 and g.daily_pnl.shape == (6, 120)
    assert all(len(d) > 0 for d in g.exit_dates)
    assert sched["n_iter"] == 6 and len(sched["windows"]) == 1
    assert 1 <= sched["windows"][0]["grid_row"] <= 6


def test_closed_pnl_equals_trade_pnl_and_daily_sum():
    text, _ = make_multiwalk(AXES, n_days=120, seed=2, structure="noise", split=80)
    g = parse_multiwalk_text(text)
    for i in range(g.n_iter):
        assert np.isclose(g.closed_pnl[i].sum(), g.exit_pnl[i].sum())
        assert np.isclose(g.daily_pnl[i].sum(), g.closed_pnl[i].sum())


def test_structures_differ_in_oos_alignment():
    xs, ys = {}, {}
    for s in ("persistent", "noise", "decay"):
        text, sched = make_multiwalk({"A": list(range(6)), "B": list(range(6))}, n_days=1200, seed=3, structure=s, split=800)
        g = parse_multiwalk_text(text)
        xs[s] = g.daily_pnl[:, :800].sum(axis=1); ys[s] = g.daily_pnl[:, 800:].sum(axis=1)
    corr = {s: np.corrcoef(xs[s], ys[s])[0, 1] for s in xs}
    assert corr["persistent"] > 0.7 and abs(corr["noise"]) < 0.4 and corr["decay"] < -0.5


def test_db_is_sqlite_with_the_tables_the_reader_needs():
    _, sched = make_multiwalk(AXES, n_days=120, seed=1, structure="persistent", split=80)
    b = make_walkforward_db(sched)
    assert b.startswith(b"SQLite format 3")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db"); tmp.write(b); tmp.close()
    try:
        con = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        assert {"WalkforwardData", "FitnessFunctions", "BacktestTimePeriodTypes", "StrategyProjects"} <= tabs
        row = con.execute("select WFTotalIterationsCount, WFWalkforwardParameterData from WalkforwardData").fetchone()
        assert row[0] == 6 and "~A,B~P|" in row[1]
        con.close()
    finally:
        os.unlink(tmp.name)
