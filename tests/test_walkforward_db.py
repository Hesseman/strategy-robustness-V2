import pandas as pd
import pytest

from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import WalkforwardDBError, parse_walkforward_db


def _db():
    _, sched = make_multiwalk({"A": [1, 2, 3], "B": [10, 20]}, n_days=120, seed=1, structure="persistent", split=80)
    return make_walkforward_db(sched), sched


def test_reads_group_periods_fitness_and_windows():
    b, sched = _db()
    groups = parse_walkforward_db(b)
    assert len(groups) == 1
    g = groups[0]
    assert g.group_no == 1 and g.strategy == "Synthetic_MW" and g.symbol == "@SYN" and g.interval == "30min"
    assert (g.in_len, g.in_type, g.out_len, g.out_type, g.anchored) == (80, "Day", 40, "Day", False)
    assert g.fitness_abbr == "NPAvgDD" and g.fitness_name == "NP/Avg DD" and g.n_iterations == 6
    assert g.param_names == ["A", "B"]
    w = g.windows[0]
    assert w.oos_start == pd.Timestamp(sched["windows"][0]["oos_start"]) and w.oos_end == pd.Timestamp(sched["windows"][0]["oos_end"])
    assert w.params == sched["windows"][0]["params"] and w.grid_row == sched["windows"][0]["grid_row"]
    assert "unanchored" in g.label and "NPAvgDD" in g.label


def test_multiwalk_schedule_string_parses_like_the_real_one():
    from robustness.walkforward_db import _parse_schedule
    s = ("v2,20260613 10:04:37,1,1096,0,0,0,1,3,10,1,0~BarToHold,TriggerProfitAmt,LockInPct"
         "~P|20200114,20230113,10,933,1096,274,5.0142|20,400,0.2|43,0,0,0|5Y-3Y,U,NPAvgDD,PMx"
         "~P|20230114,20260113,10,935,1096,274,2.0582|18,1600,0.6|231,0,0,0|5Y-3Y,U,NPAvgDD,PMx")
    names, wins = _parse_schedule(s)
    assert names == ["BarToHold", "TriggerProfitAmt", "LockInPct"]
    assert [w.grid_row for w in wins] == [43, 231] and wins[0].params == (20.0, 400.0, 0.2)
    assert wins[1].oos_start == pd.Timestamp("2023-01-14") and wins[1].label == "5Y-3Y,U,NPAvgDD,PMx"


def test_not_sqlite_is_readable_error():
    with pytest.raises(WalkforwardDBError, match="SQLite"):
        parse_walkforward_db(b"hello")


def test_missing_tables_is_readable_error(tmp_path):
    import sqlite3
    p = tmp_path / "x.db"; con = sqlite3.connect(p); con.execute("create table t(a)"); con.commit(); con.close()
    with pytest.raises(WalkforwardDBError, match="not a MultiWalk"):
        parse_walkforward_db(p.read_bytes())
