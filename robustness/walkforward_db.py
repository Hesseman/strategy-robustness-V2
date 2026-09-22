"""Read MultiWalk's WalkforwardData.db (SQLite) for the walk-forward schedule and the picks.

Only reason to change: MultiWalk changes that schema. Verified (MultiWalk Pro Apr 17 2025 build,
2026-09-22): table WalkforwardData holds one row per walk-forward group with the in/out period
lengths and types (7 Day, 8 Week, 9 Month, 10 Year), the anchored flag, the fitness-function id
(FitnessFunctions table) and the schedule string WFWalkforwardParameterData =
``v2,<stamp>,...~<param names>~P|<oos start>,<oos end>,<type>,<a>,<b>,<c>,<fitness>|<params>|<grid row>,0,0,0|<label>~P|...``
(dates yyyymmdd; grid row is 1-based into the optimisation file's iteration order). The per-window
<fitness> value is not reproduced by this app and is ignored."""
from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass, field

import pandas as pd


class WalkforwardDBError(ValueError):
    """The file is not a MultiWalk WalkforwardData.db this app can read."""


@dataclass
class WFWindow:
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    params: tuple[float, ...]
    grid_row: int
    label: str


@dataclass
class WFGroup:
    group_no: int
    strategy: str
    symbol: str
    interval: str
    in_len: int
    in_type: str
    out_len: int
    out_type: str
    anchored: bool
    fitness_abbr: str
    fitness_name: str
    n_iterations: int
    param_names: list[str]
    windows: list[WFWindow] = field(default_factory=list)

    @property
    def label(self) -> str:
        return (f"group {self.group_no}: {self.symbol} {self.interval} · {self.in_len} {self.in_type} in / "
                f"{self.out_len} {self.out_type} out, {'anchored' if self.anchored else 'unanchored'} · {self.fitness_abbr}")


_PERIOD_TYPES = {7: "Day", 8: "Week", 9: "Month", 10: "Year"}


def _parse_schedule(s: str) -> tuple[list[str], list[WFWindow]]:
    """Split WFWalkforwardParameterData into (parameter names, windows). Raises WalkforwardDBError."""
    segs = s.split("~")
    if len(segs) < 2:
        raise WalkforwardDBError("WFWalkforwardParameterData has no '~'-separated parameter-name block")
    names = [n.strip() for n in segs[1].split(",") if n.strip()]
    windows: list[WFWindow] = []
    for seg in segs[2:]:
        if not seg.startswith("P|"):
            continue
        parts = seg.split("|")
        if len(parts) < 4:
            raise WalkforwardDBError(f"window block {seg[:40]!r} has fewer than 4 '|' parts")
        d = parts[1].split(",")
        try:
            oos_start = pd.Timestamp(pd.to_datetime(d[0], format="%Y%m%d"))
            oos_end = pd.Timestamp(pd.to_datetime(d[1], format="%Y%m%d"))
            params = tuple(float(x) for x in parts[2].split(","))
            row = int(parts[3].split(",")[0])
        except (ValueError, IndexError) as e:
            raise WalkforwardDBError(f"window block {seg[:40]!r} does not parse ({e})") from None
        windows.append(WFWindow(oos_start=oos_start, oos_end=oos_end, params=params, grid_row=row,
                                label=parts[4] if len(parts) > 4 else ""))
    return names, windows


def parse_walkforward_db(db_bytes: bytes) -> list[WFGroup]:
    """Read every walk-forward group in the database.

    Accepts: the raw bytes of WalkforwardData.db.
    Returns: one WFGroup per WalkforwardData row, ordered by group number, each with its windows
    in schedule order.
    Guarantees: raises WalkforwardDBError when the bytes are not SQLite, the MultiWalk tables are
    missing, a period type is not Day/Week/Month/Year, or a schedule string does not parse; the
    temporary file used to open the bytes is removed before returning."""
    if not db_bytes.startswith(b"SQLite format 3"):
        raise WalkforwardDBError("not an SQLite file - upload WalkforwardData.db from the project's 'Walkforward Files' folder")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    try:
        tmp.write(db_bytes); tmp.close()
        con = sqlite3.connect(f"file:{tmp.name}?mode=ro", uri=True)
        try:
            ff = {r[0]: (r[1], r[2]) for r in con.execute("select FFID, FFAbbr, FFName from FitnessFunctions")}
            strat = {r[0]: r[1] for r in con.execute("select StrategyProjectID, StrategyName from StrategyProjects")}
            rows = con.execute(
                "select WFGroupNumber, WFStrategyProjectID, WFSymbol1, WFBarTypeInterval1, WFInPeriodLength, WFInPeriodType, "
                "WFOutPeriodLength, WFOutPeriodType, WFIsAnchored, WFFitnessFunctionID, WFTotalIterationsCount, "
                "WFWalkforwardParameterData from WalkforwardData order by WFGroupNumber").fetchall()
        except sqlite3.DatabaseError as e:
            raise WalkforwardDBError(f"not a MultiWalk walk-forward database ({e})") from None
        finally:
            con.close()
    finally:
        os.unlink(tmp.name)
    if not rows:
        raise WalkforwardDBError("WalkforwardData table is empty - run the walk-forward in MultiWalk first")
    groups: list[WFGroup] = []
    for r in rows:
        if r[5] not in _PERIOD_TYPES or r[7] not in _PERIOD_TYPES:
            raise WalkforwardDBError(f"group {r[0]}: unsupported period type ids {r[5]}/{r[7]} (Day/Week/Month/Year only)")
        names, windows = _parse_schedule(r[11] or "")
        abbr, name = ff.get(r[9], ("?", "unknown"))
        groups.append(WFGroup(group_no=int(r[0]), strategy=strat.get(r[1], ""), symbol=r[2] or "", interval=r[3] or "",
                              in_len=int(r[4]), in_type=_PERIOD_TYPES[r[5]], out_len=int(r[6]), out_type=_PERIOD_TYPES[r[7]],
                              anchored=bool(r[8]), fitness_abbr=abbr, fitness_name=name, n_iterations=int(r[10]),
                              param_names=names, windows=windows))
    return groups
