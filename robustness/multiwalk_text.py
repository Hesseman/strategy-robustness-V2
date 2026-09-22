"""Parse MultiWalk's legacy text optimization file (``…_MultiWalk.txt``).

Only reason to change: MultiWalk changes that layout. Verified layout (MultiWalk Pro, Apr 17
2025 build, LE2601 MNQ 30-min project, 2026-09-22): line 1 = the optimized input names,
comma-separated; then one line per iteration in MultiWalk's grid-row order (first input varies
fastest):

    <param values>~v1,<n>|<daily>|<daily>|...~v2|<trade>|<trade>|...

daily = ``yyyymmdd,m2m_pnl,f3,f4,f5,f6,f7,minutes_in_market,closed_pnl`` (one per trading day of
the whole optimisation period, identical dates on every line); trade = ``yyyymmdd,hhmm,E|X,L|S,
price,pnl,mae,mfe``. Only the date, the mark-to-market P&L, the closed P&L and the exit rows are
used. The file exists only when the project's MultiWalkSetup.txt carries
``iUseLegacyOptimizationTextFileFormat: true`` (otherwise MultiWalk writes a sealed ``.dat``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class MultiWalkFormatError(ValueError):
    """The text is not a MultiWalk legacy optimization file this app can read."""


@dataclass
class MultiWalkGrid:
    """One optimisation run: every parameter combination's daily P&L and closed trades.

    param_names: the optimized inputs in file order; params: (n_iter, k) values per iteration
    in file order (MultiWalk grid row r is params[r - 1]); axes: sorted distinct values per
    input; grid_pos: (n_iter, k) position of each iteration on each axis; dates: the trading
    days shared by every iteration; daily_pnl / closed_pnl: (n_iter, n_days) $ per day
    (mark-to-market / closed trades); exit_dates / exit_pnl: per iteration, one entry per
    closed trade (datetime64[D] / $)."""
    param_names: list[str]
    params: np.ndarray
    axes: list[np.ndarray]
    grid_pos: np.ndarray
    dates: pd.DatetimeIndex
    daily_pnl: np.ndarray
    closed_pnl: np.ndarray
    exit_dates: list[np.ndarray]
    exit_pnl: list[np.ndarray]

    @property
    def n_iter(self) -> int:
        return int(self.params.shape[0])

    @property
    def n_days(self) -> int:
        return int(len(self.dates))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    @property
    def is_full_grid(self) -> bool:
        distinct = len({tuple(p) for p in self.grid_pos.tolist()})
        return int(np.prod(self.shape)) == self.n_iter == distinct


def _ymd(s: str) -> str:
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _split_iteration(line: str, k: int, li: int) -> tuple[list[float], list[str], list[str]]:
    segs = line.split("~")
    if len(segs) < 2:
        raise MultiWalkFormatError(f"iteration {li}: expected '<params>~v1,...|...' but found no '~' separator")
    try:
        params = [float(x) for x in segs[0].split(",")]
    except ValueError:
        raise MultiWalkFormatError(f"iteration {li}: parameter values {segs[0]!r} are not numbers") from None
    if len(params) != k:
        raise MultiWalkFormatError(f"iteration {li}: {len(params)} parameter values for {k} names")
    daily = segs[1].split("|")
    if not daily[0].startswith("v1"):
        raise MultiWalkFormatError(f"iteration {li}: daily block does not start with 'v1' (found {daily[0]!r})")
    trades: list[str] = []
    if len(segs) > 2:
        tr = segs[2].split("|")
        if not tr[0].startswith("v2"):
            raise MultiWalkFormatError(f"iteration {li}: trade block does not start with 'v2' (found {tr[0]!r})")
        trades = [t for t in tr[1:] if t]
    return params, [d for d in daily[1:] if d], trades


def parse_multiwalk_text(text: str) -> MultiWalkGrid:
    """Parse the legacy text optimization file.

    Accepts: the file's text (BOM tolerated, CRLF or LF).
    Returns: a MultiWalkGrid (see the dataclass) in file order.
    Guarantees: raises MultiWalkFormatError naming the first bad iteration when the header is
    missing, a value does not parse, a block tag is missing, a daily record has fewer than 9
    fields, or an iteration's dates differ from iteration 1's; dates are strictly increasing;
    daily_pnl.shape == closed_pnl.shape == (n_iter, n_days); an iteration without a v2 block
    has zero closed trades."""
    lines = [l.strip() for l in text.lstrip("﻿").replace("\r\n", "\n").split("\n")]
    lines = [l for l in lines if l]
    if len(lines) < 2 or "~" in lines[0]:
        raise MultiWalkFormatError("no header line with the optimized input names - is this the _MultiWalk.txt "
                                   "written with iUseLegacyOptimizationTextFileFormat: true?")
    names = [n.strip() for n in lines[0].split(",")]
    k = len(names)
    params: list[list[float]] = []
    daily_rows: list[list[float]] = []
    closed_rows: list[list[float]] = []
    exit_dates: list[np.ndarray] = []
    exit_pnl: list[np.ndarray] = []
    ref_dates: list[str] | None = None
    for li, line in enumerate(lines[1:], 1):
        p, daily, trades = _split_iteration(line, k, li)
        cells = [d.split(",") for d in daily]
        if not cells or any(len(c) < 9 for c in cells):
            raise MultiWalkFormatError(f"iteration {li}: a daily record has fewer than 9 fields")
        dates = [c[0] for c in cells]
        if ref_dates is None:
            ref_dates = dates
        elif dates != ref_dates:
            raise MultiWalkFormatError(f"iteration {li}: its trading days differ from iteration 1")
        try:
            daily_rows.append([float(c[1]) for c in cells])
            closed_rows.append([float(c[8]) for c in cells])
            ex = [t.split(",") for t in trades]
            ex = [t for t in ex if len(t) >= 6 and t[2] == "X"]
            exit_dates.append(np.array([_ymd(t[0]) for t in ex], dtype="datetime64[D]"))
            exit_pnl.append(np.array([float(t[5]) for t in ex], dtype=float))
        except ValueError as e:
            raise MultiWalkFormatError(f"iteration {li}: a numeric field does not parse ({e})") from None
        params.append(p)
    dates_idx = pd.DatetimeIndex(pd.to_datetime(ref_dates, format="%Y%m%d"))
    if not dates_idx.is_monotonic_increasing or dates_idx.has_duplicates:
        raise MultiWalkFormatError("daily records are not in strictly increasing date order")
    params_arr = np.asarray(params, dtype=float)
    axes = [np.unique(params_arr[:, j]) for j in range(k)]
    grid_pos = np.column_stack([np.searchsorted(axes[j], params_arr[:, j]) for j in range(k)]).astype(int)
    return MultiWalkGrid(param_names=names, params=params_arr, axes=axes, grid_pos=grid_pos, dates=dates_idx,
                         daily_pnl=np.asarray(daily_rows, dtype=float), closed_pnl=np.asarray(closed_rows, dtype=float),
                         exit_dates=exit_dates, exit_pnl=exit_pnl)
