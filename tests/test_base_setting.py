"""The recommended base setting (docs/wfc-region-lift.md, 'Base setting'): the centre of the
largest connected top-20% region of the pooled surface over the basis, a spread ensemble, a
confidence taken from the WFC verdict, and Kaufman's top-5 average for display only."""
import numpy as np
import pandas as pd
import pytest

from robustness.base_setting import base_setting, basis_mask, classify_axis
from robustness.multiwalk_text import MultiWalkGrid
from robustness.windows import Window


def test_parameters_are_classified_by_name_scale_free_words_winning():
    expected = {
        "TriggerProfitAmt": "price-scaled", "StopLossAmt": "price-scaled", "ProfitTargetAmt": "price-scaled",
        "StopLossDollars": "price-scaled", "TgtDolB": "price-scaled", "stopl": "price-scaled", "ptarget": "price-scaled",
        "SL": "price-scaled", "TP": "price-scaled", "cStop": "price-scaled", "ProfitTarg": "price-scaled",
        "LockInPct": "scale-free", "BarToHold": "scale-free", "ExitBars": "scale-free", "Length1": "scale-free",
        "EMAPeriod": "scale-free", "VolumeLookback": "scale-free", "numATR": "scale-free", "fac": "scale-free",
        "ProfitTargetMultiplier": "scale-free", "PercentX": "scale-free", "RSITriggerLow": "scale-free",
        "ADXThreashold": "scale-free", "EndTimeUP": "scale-free", "StopATR": "scale-free",
        "InputVar1": "unclassified", "ncons": "unclassified", "BreakEvenX": "unclassified", "ncontracts": "unclassified",
    }
    assert {n: classify_axis(n) for n in expected} == expected


def _grid(daily, names=("A",)):
    """A one-parameter line grid (index = position) from a (cells x days) daily $ P&L matrix."""
    daily = np.asarray(daily, dtype=float)
    n, t = daily.shape
    dates = pd.bdate_range("2021-01-04", periods=t)
    pos = np.arange(n)[:, None]
    return MultiWalkGrid(param_names=list(names), params=pos.astype(float) * 100.0, axes=[np.arange(n, dtype=float) * 100.0],
                         grid_pos=pos, dates=dates, daily_pnl=daily, closed_pnl=daily.copy(),
                         exit_dates=[dates.values[daily[i] != 0].astype("datetime64[D]") for i in range(n)],
                         exit_pnl=[daily[i][daily[i] != 0] for i in range(n)])


def _window(grid, index, is_days, oos_days, complete=True):
    days = np.arange(grid.n_days)
    d = grid.dates
    return Window(index=index, label=f"w{index}", is_start=d[is_days[0]], is_end=d[is_days[-1]], oos_start=d[oos_days[0]],
                  oos_end_nominal=d[oos_days[-1]], oos_end=d[oos_days[-1]], is_mask=np.isin(days, is_days),
                  oos_mask=np.isin(days, oos_days), complete=complete, grid_row=1, params=(0.0,))


def test_the_basis_is_the_last_windows_in_and_out_of_sample_or_the_later_half_on_one_split():
    grid = _grid(np.zeros((3, 30)))
    windows = [_window(grid, 1, list(range(0, 10)), list(range(10, 20))),
               _window(grid, 2, list(range(10, 20)), list(range(20, 30)), complete=False)]
    assert np.flatnonzero(basis_mask(windows, grid.n_days, "multiwalk")).tolist() == list(range(10, 30))
    assert np.flatnonzero(basis_mask(windows, grid.n_days, "two")).tolist() == list(range(10, 30))
    one = [_window(grid, 1, list(range(0, 15)), list(range(15, 30)))]
    assert np.flatnonzero(basis_mask(one, grid.n_days, "single")).tolist() == list(range(15, 30))


# Hand trace, ten cells in a line, basis = days 0-3 (one window, IS 0-1, OOS 2-3), all P&L on day 0:
#   basis net profit  [0, 0, 0, 6, 9, 6, 0, 0, 0, 0]
#   pooled            [0, 0, 2, 5, 7, 5, 2, 0, 0, 0]   -> top-fifth threshold 5 -> region {3, 4, 5}
#   centre            cell 4 (centroid); k = min(4, 3 // 3) = 1 -> ensemble [4]
#   Kaufman's five    cells 4, 3, 5, 0, 1 (ties in grid order) -> mean position 2.6 -> cell 3; not one area
_BASIS = [[0.0] * 4 for _ in range(10)]
for _i, _v in ((3, 6.0), (4, 9.0), (5, 6.0)):
    _BASIS[_i][0] = _v


def test_base_setting_hand_traced():
    grid = _grid(_BASIS, names=("BarToHold",))
    w = _window(grid, 1, [0, 1], [2, 3])
    b = base_setting(grid, [w], "edge", "multiwalk")
    assert b.confidence == "supported" and b.reading == "edge"
    assert (b.centre, b.centre_is_peak, b.ensemble, b.n_component) == (4, False, [4], 3)
    assert b.ranges == [(300.0, 500.0)] and b.axis_class == ["scale-free"]
    assert (b.kaufman, b.kaufman_contiguous) == (3, False)
    assert (b.basis_start, b.basis_end, b.n_basis_days) == (grid.dates[0], grid.dates[3], 4)
    assert b.pooled.tolist() == pytest.approx([0, 0, 2, 5, 7, 5, 2, 0, 0, 0])


def test_confidence_follows_the_verdict():
    grid = _grid(_BASIS)
    w = _window(grid, 1, [0, 1], [2, 3])
    for reading, confidence in (("edge", "supported"), ("plateau", "low stakes"), ("noise", "none"),
                                ("loser", "none"), ("insufficient", "none")):
        assert base_setting(grid, [w], reading, "multiwalk").confidence == confidence
