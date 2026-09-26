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


def test_planted_grid_default_stream_is_pinned():
    """The default Gaussian stream feeds every size and power oracle of the WFC card; captured before
    the tail option existed, so adding one cannot move it."""
    from robustness.synthetic_multiwalk import make_planted_grid
    g = make_planted_grid((3, 3), structure="noise", rho=0.6, trade_p=0.5, n_days=40, seed=4)
    assert g.daily_pnl[0, :8].tolist() == [0.0, -0.0, -0.0, 35.46, 0.0, 89.35, 0.0, -103.81]
    assert g.daily_pnl[8, -6:].tolist() == [34.23, 0.0, -22.32, 0.0, 62.78, -101.75]


def test_jumps_tail_is_rare_large_winners_centred_on_zero():
    """tail='jumps' (rare large trades): each trade's noise is a large winner with probability 0.03 and a
    small loser otherwise, standardised to mean 0 and SD 1. With rho = 0 (no shared field) a 'noise'
    grid's trades take exactly two values, 100 x 0.97 / sqrt(0.03 x 0.97) = +568.62 and
    100 x -0.03 / sqrt(0.03 x 0.97) = -17.59, the winners about 3% of the trades."""
    import pytest
    from robustness.synthetic_multiwalk import make_planted_grid
    g = make_planted_grid((6, 6), structure="noise", rho=0.0, trade_p=0.5, n_days=600, tail="jumps", seed=0)
    trades = g.daily_pnl[:, g.daily_pnl.any(axis=0)]
    assert set(np.unique(trades).tolist()) == {568.62, -17.59}
    assert 0.02 <= (trades > 0).mean() <= 0.04
    with pytest.raises(ValueError):
        make_planted_grid((3, 3), tail="cauchy")
