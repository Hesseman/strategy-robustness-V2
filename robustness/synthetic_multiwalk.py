"""Synthetic MultiWalk outputs for tests and the demo button: a legacy text optimisation file
and a matching WalkforwardData.db, written in the exact layouts that multiwalk_text and
walkforward_db read. Structure 'persistent' plants a smooth bump on the grid that is the same
in-sample and out-of-sample; 'noise' plants nothing; 'decay' flips the bump's sign after the
split. Trading days are weekdays from 2020-01-06. make_planted_grid builds a MultiWalkGrid directly
(no text round trip) with correlated neighbour noise, for the null-size and power oracles."""
from __future__ import annotations

import itertools
import os
import sqlite3
import tempfile

import numpy as np
import pandas as pd

from robustness.multiwalk_text import MultiWalkGrid

_PERIOD_DAY = 7
_FITNESS_NPAVGDD = 6


def _bump(pos: np.ndarray, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    center = np.array([rng.uniform(0.3, 0.7) * (s - 1) for s in shape])
    width = np.array([max(1.0, 0.6 * (s - 1)) for s in shape])
    d2 = (((pos - center) / width) ** 2).sum(axis=1)
    return np.exp(-d2) - 0.45  # positive near the centre, negative at the rim


def make_multiwalk(axes: dict[str, list[float]], n_days: int = 600, seed: int = 0, structure: str = "persistent",
                   split: int | None = None, signal: float = 40.0, noise: float = 100.0,
                   trade_every: int = 7) -> tuple[str, dict]:
    """Build a synthetic optimisation.

    Accepts: axes - ordered {input name: sorted values} (the first input varies fastest, as
    MultiWalk writes it); n_days trading days; seed; structure in {'persistent','noise','decay'};
    split - index of the first out-of-sample day (default 2/3 of n_days); signal / noise - daily $
    scale of the planted bump and of the noise; trade_every - days per closed trade.
    Returns: (text, schedule) - the legacy text file content and a schedule dict for
    make_walkforward_db with ONE window (IS = days before split, OOS = days from split) whose
    pick is the iteration with the highest in-sample net profit.
    Guarantees: parse_multiwalk_text(text) reproduces the grid; each iteration's closed P&L
    equals its trade P&L and its daily P&L sums to the same total."""
    if structure not in ("persistent", "noise", "decay"):
        raise ValueError("structure must be persistent, noise or decay")
    rng = np.random.default_rng(seed)
    names = list(axes)
    values = [list(map(float, axes[n])) for n in names]
    shape = tuple(len(v) for v in values)
    combos = list(itertools.product(*[range(s) for s in reversed(shape)]))  # last axis outermost -> first fastest
    pos = np.array([tuple(reversed(c)) for c in combos], dtype=int)
    n_iter = len(pos)
    split = int(n_days * 2 / 3) if split is None else int(split)
    dates = pd.bdate_range("2020-01-06", periods=n_days)
    bump = _bump(pos, shape, rng) if structure != "noise" else np.zeros(n_iter)
    factor = np.ones(n_days)
    if structure == "decay":
        factor[split:] = -1.0
    market = rng.normal(0.0, noise * 0.15, size=n_days)   # a common market term; kept small so it cannot flip every cell's OOS sign
    idio = rng.normal(0.0, noise, size=(n_iter, n_days))
    daily = idio + market[None, :] + signal * bump[:, None] * factor[None, :]
    daily = np.round(daily, 2)
    lines = [",".join(names)]
    for i in range(n_iter):
        pvals = [values[j][pos[i, j]] for j in range(len(names))]
        closed = np.zeros(n_days)
        trades = []
        for start in range(0, n_days, trade_every):
            end = min(n_days, start + trade_every) - 1
            pnl = round(float(daily[i, start:end + 1].sum()), 2)
            closed[end] = pnl
            d0, d1 = dates[start].strftime("%Y%m%d"), dates[end].strftime("%Y%m%d")
            trades.append(f"{d0},1000,E,L,100.0,0,0,0")
            trades.append(f"{d1},1500,X,L,100.0,{pnl},{min(pnl, 0.0)},{max(pnl, 0.0)}")
        recs = [f"{dates[t].strftime('%Y%m%d')},{daily[i, t]:.2f},0,0,1,1,0,1440,{closed[t]:.2f}" for t in range(n_days)]
        lines.append(",".join(f"{v:g}" for v in pvals) + "~v1,1|" + "|".join(recs) + "~v2|" + "|".join(trades))
    is_np = daily[:, :split].sum(axis=1)
    best = int(np.argmax(is_np))
    schedule = {"param_names": names, "n_iter": n_iter, "in_len": split, "in_type_id": _PERIOD_DAY,
                "out_len": n_days - split, "out_type_id": _PERIOD_DAY, "anchored": False, "fitness_id": _FITNESS_NPAVGDD,
                "symbol": "@SYN", "interval": "30min", "strategy": "Synthetic_MW",
                "windows": [{"oos_start": dates[split].strftime("%Y%m%d"), "oos_end": dates[-1].strftime("%Y%m%d"),
                             "params": tuple(values[j][pos[best, j]] for j in range(len(names))), "grid_row": best + 1}]}
    return "\n".join(lines) + "\n", schedule


PLANTED = ("noise", "ridge", "persistent", "decay")


def make_planted_grid(shape: tuple[int, ...] = (6, 6), *, structure: str = "noise", rho: float = 0.9, trade_p: float = 0.2,
                      n_days: int = 600, split: int | None = None, signal: float = 0.3, seed: int = 0) -> MultiWalkGrid:
    """A planted-truth optimisation grid whose neighbouring cells share most of their noise, as real
    neighbouring parameter sets share most of their trades (the design behind docs/wfc-region-lift.md,
    'Evidence'). Every cell trades on the same days (probability trade_p per day); per trade
    $ P&L = 100 x (mu(cell) x factor(day) + noise), noise = sqrt(rho) x a field smoothed over the grid
    (Chebyshev radius 2) + sqrt(1 - rho) x the cell's own noise; rounded to cents.

    Accepts: shape; structure in PLANTED - 'noise' (mu = 0), 'ridge' (a narrow bump covering about a
    fifth of an otherwise flat grid), 'persistent' (a wide bump, positive centre and negative rim) or
    'decay' (the wide bump with its sign flipped from day `split`, default 2/3 of n_days); rho in
    [0, 1]; trade_p; n_days; signal = mu at the bump's peak, in units of the noise SD; seed.
    Returns: a MultiWalkGrid on weekdays from 2020-01-06 (last axis varies fastest; params = grid
    positions); closed P&L = daily P&L, one exit per trading day.
    Guarantees: deterministic for a seed; a 'noise' grid does not depend on signal or split; raises
    ValueError for another structure."""
    if structure not in PLANTED:
        raise ValueError(f"structure must be one of {PLANTED}, got {structure!r}")
    rng = np.random.default_rng(seed)
    pos = np.array(list(itertools.product(*[range(s) for s in shape])), dtype=int)
    n = len(pos)
    trade_day = rng.random(n_days) < trade_p
    white = rng.normal(size=(n, n_days))
    d = np.abs(pos[:, None, :] - pos[None, :, :]).max(axis=2)
    field = np.stack([white[d[i] <= 2].mean(axis=0) for i in range(n)])
    field /= field.std(axis=1, keepdims=True)
    noise = np.sqrt(rho) * field + np.sqrt(1 - rho) * rng.normal(size=(n, n_days))
    mu, factor = np.zeros(n), np.ones(n_days)
    if structure != "noise":   # the bump's centre comes from its own stream, so the noise above is shared by every structure
        centre = np.array([np.random.default_rng([seed, 1]).uniform(0.3, 0.7) * (s - 1) for s in shape])
        if structure == "ridge":
            width = np.array([max(0.7, 0.28 * (s - 1)) for s in shape])
            mu = signal * np.exp(-(((pos - centre) / width) ** 2).sum(axis=1))
        else:
            width = np.array([max(1.0, 0.6 * (s - 1)) for s in shape])
            mu = signal * (np.exp(-(((pos - centre) / width) ** 2).sum(axis=1)) - 0.45)
        if structure == "decay":
            factor[(int(n_days * 2 / 3) if split is None else int(split)):] = -1.0
    daily = np.round(100.0 * trade_day[None, :] * (mu[:, None] * factor[None, :] + noise), 2)
    dates = pd.bdate_range("2020-01-06", periods=n_days)
    ed = dates.values[trade_day].astype("datetime64[D]")
    return MultiWalkGrid(param_names=[f"p{k}" for k in range(len(shape))], params=pos.astype(float),
                         axes=[np.arange(s, dtype=float) for s in shape], grid_pos=pos, dates=dates, daily_pnl=daily,
                         closed_pnl=daily.copy(), exit_dates=[ed] * n, exit_pnl=[daily[i, trade_day] for i in range(n)])


def make_walkforward_db(schedule: dict) -> bytes:
    """Write a minimal WalkforwardData.db (the tables and columns walkforward_db reads) for a
    schedule dict from make_multiwalk. Returns the SQLite file's bytes; the temp file is removed."""
    names = ",".join(schedule["param_names"])
    blocks = []
    for w in schedule["windows"]:
        params = ",".join(f"{v:g}" for v in w["params"])
        blocks.append(f"P|{w['oos_start']},{w['oos_end']},{schedule['out_type_id']},0,0,0,0.0000|{params}|{w['grid_row']},0,0,0|synthetic")
    sched = f"v2,20260101 00:00:00,1,0,0,0,0,1,{schedule['out_len']},{schedule['out_type_id']},1,0~{names}~" + "~".join(blocks)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db"); tmp.close()
    try:
        con = sqlite3.connect(tmp.name)
        con.executescript("""
            CREATE TABLE FitnessFunctions (FFID INTEGER PRIMARY KEY, FFAbbr TEXT, FFName TEXT, FFDesc TEXT);
            INSERT INTO FitnessFunctions VALUES (1,'NP','Net Profit',''),(6,'NPAvgDD','NP/Avg DD',''),(21,'Sharpe','Sharpe Ratio','');
            CREATE TABLE BacktestTimePeriodTypes (BacktestTimePeriodTypeID INTEGER PRIMARY KEY, BacktestTimePeriodTypeDesc TEXT);
            INSERT INTO BacktestTimePeriodTypes VALUES (7,'Day'),(8,'Week'),(9,'Month'),(10,'Year');
            CREATE TABLE StrategyProjects (StrategyProjectID INTEGER PRIMARY KEY, StrategyName TEXT);
            CREATE TABLE WalkforwardData (WalkforwardID INTEGER PRIMARY KEY, WFGroupNumber INTEGER, WFStrategyProjectID INTEGER,
              WFSymbol1 TEXT, WFBarTypeInterval1 TEXT, WFInPeriodLength INTEGER, WFInPeriodType INTEGER, WFOutPeriodLength INTEGER,
              WFOutPeriodType INTEGER, WFIsAnchored INTEGER, WFFitnessFunctionID INTEGER, WFTotalIterationsCount INTEGER,
              WFWalkforwardParameterData TEXT);
        """)
        con.execute("INSERT INTO StrategyProjects VALUES (1, ?)", (schedule["strategy"],))
        con.execute("INSERT INTO WalkforwardData VALUES (1,1,1,?,?,?,?,?,?,?,?,?,?)",
                    (schedule["symbol"], schedule["interval"], schedule["in_len"], schedule["in_type_id"], schedule["out_len"],
                     schedule["out_type_id"], int(schedule["anchored"]), schedule["fitness_id"], schedule["n_iter"], sched))
        con.commit(); con.close()
        return open(tmp.name, "rb").read()
    finally:
        os.unlink(tmp.name)
