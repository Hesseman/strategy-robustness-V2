"""The WFC card's 'Read with the PASS' line (docs/wfc-region-lift.md, 'Printed guards'): the facts a
PASS is read with, including the lift's p under both nulls and a flagged disagreement."""
from app.wfc_text import read_with_the_pass

_GUARDS = {"n_eff_median": 1.3, "few_variants": True, "region_oos_mean": 12000.0, "grid_oos_mean": 5000.0,
           "region_minus_grid": [9180.0, -2000.0], "windows_ahead": 1, "windows_complete": 2,
           "p_lift_boot": 0.041, "nulls_disagree": False}


def test_the_pass_line_prints_the_lift_p_under_both_nulls():
    line = read_with_the_pass(_GUARDS, p_flip=0.032)
    assert line.startswith("**Read with the PASS:** ")
    assert "p = 0.032 under the sign-flip null (the gate)" in line
    assert "p = 0.041 under the centred block bootstrap (its cross-check; the two agree)" in line
    assert "disagree" not in line.replace("the two agree", "")
    assert "≈ **1.3** effective independent variants - fewer than 3, so the lift rests on the handful of trades" in line
    assert ("out of sample the region made **$12,000** against the grid's **$5,000** (+$7,000); ahead of the grid in "
            "**1 of 2** complete window(s): +$9,180, −$2,000") in line


def test_a_disagreement_between_the_nulls_is_flagged():
    line = read_with_the_pass({**_GUARDS, "p_lift_boot": 0.071, "nulls_disagree": True}, p_flip=0.032)
    assert "p = 0.032 under the sign-flip null (the gate) but p = 0.071 under the centred block bootstrap" in line
    assert "**the two nulls disagree**" in line and "symmetric" in line


def test_without_n_eff_the_line_still_prints_the_nulls_and_the_dollars():
    line = read_with_the_pass({**_GUARDS, "n_eff_median": float("nan"), "few_variants": False}, p_flip=0.0005)
    assert "p = 0.0005 under the sign-flip null" in line and "effective independent variants" not in line
    assert "ahead of the grid in **1 of 2**" in line
