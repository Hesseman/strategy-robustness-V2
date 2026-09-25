import numpy as np
import pandas as pd

from robustness.walkforward_db import WFGroup, WFWindow
from robustness.windows import SCHEMES, custom_windows, derive_windows


def _group(anchored=False, in_type="Year", in_len=5):
    wins = [WFWindow(pd.Timestamp("2020-01-14"), pd.Timestamp("2023-01-13"), (20.0, 400.0, 0.2), 43, ""),
            WFWindow(pd.Timestamp("2023-01-14"), pd.Timestamp("2026-01-13"), (18.0, 1600.0, 0.6), 231, ""),
            WFWindow(pd.Timestamp("2026-01-14"), pd.Timestamp("2029-01-13"), (19.0, 1200.0, 0.6), 222, "")]
    return WFGroup(1, "LE", "@MNQ", "30min", in_len, in_type, 3, "Year", anchored, "NPAvgDD", "NP/Avg DD", 240,
                   ["BarToHold", "TriggerProfitAmt", "LockInPct"], wins)


DATES = pd.bdate_range("2015-01-14", "2026-09-21")


def test_unanchored_five_year_windows_like_le2601():
    w = derive_windows(_group(), DATES)
    assert [x.index for x in w] == [1, 2, 3]
    assert (w[0].is_start, w[0].is_end, w[0].oos_start, w[0].oos_end) == (
        pd.Timestamp("2015-01-14"), pd.Timestamp("2020-01-13"), pd.Timestamp("2020-01-14"), pd.Timestamp("2023-01-13"))
    assert (w[1].is_start, w[1].is_end) == (pd.Timestamp("2018-01-14"), pd.Timestamp("2023-01-13"))
    assert w[0].complete and w[1].complete and not w[2].complete
    assert w[2].oos_end == pd.Timestamp("2026-09-21") and w[2].oos_end_nominal == pd.Timestamp("2029-01-13")
    assert w[0].is_mask.sum() > 1200 and w[0].oos_mask.sum() > 700 and w[0].grid_row == 43
    assert not (w[0].is_mask & w[0].oos_mask).any()
    assert DATES[w[0].oos_mask][0] == pd.Timestamp("2020-01-14") and DATES[w[0].is_mask][-1] == pd.Timestamp("2020-01-13")
    assert "incomplete" in w[2].label and "incomplete" not in w[0].label


def test_anchored_is_starts_at_first_day():
    w = derive_windows(_group(anchored=True), DATES)
    assert w[1].is_start == DATES[0] and w[1].is_end == pd.Timestamp("2023-01-13")


def test_month_periods_and_completeness_threshold():
    g = _group(in_type="Month", in_len=18)
    w = derive_windows(g, DATES, min_complete_frac=0.2)
    assert w[0].is_start == pd.Timestamp("2018-07-14")
    assert w[2].complete  # 2026-01-14..2026-09-21 is 23% of the nominal 3 years


def test_custom_single_split_is_two_halves():
    dates = pd.bdate_range("2020-01-06", periods=101)
    (w,) = custom_windows(dates, "single")
    assert w.complete and w.grid_row == 0 and w.params == ()
    assert w.is_mask.sum() == 50 and w.oos_mask.sum() == 51 and not (w.is_mask & w.oos_mask).any()
    assert w.is_mask[:50].all() and w.oos_mask[50:].all()
    assert w.label.startswith("split: IS 2020-01-06") and "OOS" in w.label
    assert w.oos_start == dates[50] and w.oos_end == dates[-1]


def test_custom_two_windows_roll_over_equal_thirds():
    dates = pd.bdate_range("2020-01-06", periods=90)
    w1, w2 = custom_windows(dates, "two")
    assert [w.index for w in (w1, w2)] == [1, 2] and w1.complete and w2.complete
    assert w1.is_mask[:30].all() and w1.is_mask.sum() == 30 and w1.oos_mask[30:60].all() and w1.oos_mask.sum() == 30
    assert np.array_equal(w2.is_mask, w1.oos_mask) and w2.oos_mask[60:].all() and w2.oos_mask.sum() == 30
    assert w1.label.startswith("window 1: IS") and w2.label.startswith("window 2: IS")


def test_custom_windows_reject_bad_input():
    dates = pd.bdate_range("2020-01-06", periods=90)
    assert SCHEMES == ("multiwalk", "two", "single")
    for bad in ("multiwalk", "three", ""):
        try:
            custom_windows(dates, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted")
    try:
        custom_windows(dates[:5], "single")
    except ValueError:
        return
    raise AssertionError("5 days accepted")