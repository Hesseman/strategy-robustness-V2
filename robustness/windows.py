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
