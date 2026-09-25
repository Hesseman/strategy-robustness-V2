"""Each walk-forward window's in-sample and out-of-sample date ranges (spec decision 3).
OOS = [start, min(end, last data day)]; IS = unanchored: [OOS start - IN period, OOS start - 1 day],
anchored: [first data day, OOS start - 1 day]. Complete when the actual OOS span is at least
min_complete_frac of the nominal span."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from robustness.walkforward_db import WFGroup

_OFFSET = {"Day": lambda n: pd.DateOffset(days=n), "Week": lambda n: pd.DateOffset(weeks=n),
           "Month": lambda n: pd.DateOffset(months=n), "Year": lambda n: pd.DateOffset(years=n)}
# 'multiwalk' = the DB schedule and MultiWalk's picks; 'two' / 'single' = custom_windows below
SCHEMES = ("multiwalk", "two", "single")


@dataclass
class Window:
    index: int
    label: str
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end_nominal: pd.Timestamp
    oos_end: pd.Timestamp
    is_mask: np.ndarray
    oos_mask: np.ndarray
    complete: bool
    grid_row: int
    params: tuple[float, ...]

    @property
    def n_is_days(self) -> int:
        return int(self.is_mask.sum())

    @property
    def n_oos_days(self) -> int:
        return int(self.oos_mask.sum())


def derive_windows(group: WFGroup, dates: pd.DatetimeIndex, *, min_complete_frac: float = 0.5) -> list[Window]:
    """Derive the IS/OOS masks over `dates` for every window of `group`.

    Accepts: a WFGroup (periods, anchored flag, windows) and the optimisation's trading days.
    Returns: one Window per schedule window, 1-based index, in schedule order, with boolean
    masks over `dates`.
    Guarantees: is_mask and oos_mask never overlap; oos_end <= dates[-1]; complete is False when
    the OOS mask is empty or the actual span (days, inclusive) is below min_complete_frac of the
    nominal span; labels of incomplete windows end with '(incomplete)'."""
    out: list[Window] = []
    one_day = pd.Timedelta(days=1)
    for i, w in enumerate(group.windows, 1):
        is_end = w.oos_start - one_day
        is_start = dates[0] if group.anchored else w.oos_start - _OFFSET[group.in_type](group.in_len)
        oos_end = min(w.oos_end, dates[-1])
        nominal = (w.oos_end - w.oos_start).days + 1
        actual = (oos_end - w.oos_start).days + 1
        is_mask = np.asarray((dates >= is_start) & (dates <= is_end), dtype=bool)
        oos_mask = np.asarray((dates >= w.oos_start) & (dates <= oos_end), dtype=bool)
        complete = bool(oos_mask.any() and actual >= min_complete_frac * nominal)
        label = f"window {i}: OOS {w.oos_start:%Y-%m-%d} → {oos_end:%Y-%m-%d}" + ("" if complete else " (incomplete)")
        out.append(Window(index=i, label=label, is_start=pd.Timestamp(is_start), is_end=pd.Timestamp(is_end),
                          oos_start=w.oos_start, oos_end_nominal=w.oos_end, oos_end=pd.Timestamp(oos_end),
                          is_mask=is_mask, oos_mask=oos_mask, complete=complete, grid_row=w.grid_row, params=w.params))
    return out


def custom_windows(dates: pd.DatetimeIndex, scheme: str) -> list[Window]:
    """Windows that ignore MultiWalk's schedule, for strategies with too few trades per window.

    Accepts: the optimisation's trading days and scheme 'single' (one split: in-sample = the first
    half of the days, out-of-sample = the second half) or 'two' (three equal blocks of days: IS
    block 1 -> OOS block 2, IS block 2 -> OOS block 3; unanchored, equal lengths).
    Returns: complete Windows with 1-based index, grid_row 0 and params () - the caller sets the
    pick (the best in-sample variant) once the metric is known.
    Guarantees: IS and OOS masks never overlap and together cover their blocks exactly; fixed
    rules, no tunable split; raises ValueError for another scheme or fewer than 6 trading days."""
    if scheme not in ("single", "two"):
        raise ValueError(f"scheme must be 'single' or 'two', got {scheme!r}")
    n = len(dates)
    if n < 6:
        raise ValueError(f"need at least 6 trading days, got {n}")
    cuts = [0, n // 2, n] if scheme == "single" else [0, n // 3, 2 * n // 3, n]
    out: list[Window] = []
    for i in range(1, len(cuts) - 1):
        a, b, c = cuts[i - 1], cuts[i], cuts[i + 1]
        is_mask, oos_mask = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        is_mask[a:b], oos_mask[b:c] = True, True
        prefix = "split" if scheme == "single" else f"window {i}"
        label = f"{prefix}: IS {dates[a]:%Y-%m-%d} → {dates[b - 1]:%Y-%m-%d}, OOS {dates[b]:%Y-%m-%d} → {dates[c - 1]:%Y-%m-%d}"
        out.append(Window(index=i, label=label, is_start=pd.Timestamp(dates[a]), is_end=pd.Timestamp(dates[b - 1]),
                          oos_start=pd.Timestamp(dates[b]), oos_end_nominal=pd.Timestamp(dates[c - 1]),
                          oos_end=pd.Timestamp(dates[c - 1]), is_mask=is_mask, oos_mask=oos_mask, complete=True,
                          grid_row=0, params=()))
    return out
