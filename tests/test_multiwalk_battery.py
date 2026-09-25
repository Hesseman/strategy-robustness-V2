import json

import numpy as np
import pytest

from robustness.multiwalk_battery import (CAVEAT_MW, NEFF_MIN, READINGS, MultiWalkValidationFailed, mw_to_json, region_reading,
                                          run_multiwalk_battery)
from robustness.multiwalk_text import parse_multiwalk_text
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db

AX = {"A": list(range(6)), "B": list(range(6))}


def _inputs(structure, seed=21, n_days=1200):
    text, sched = make_multiwalk(AX, n_days=n_days, seed=seed, structure=structure, split=int(n_days * 2 / 3))
    return parse_multiwalk_text(text), parse_walkforward_db(make_walkforward_db(sched)), sched


def test_persistent_passes_wfc_gate_and_reports_everything():
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=100, seed=0)
    assert r.verdicts == {"wfc": "pass", "plateau": "score", "selection": "reference"}
    assert r.gates_passed == 1 and r.gates_total == 1 and r.caveat == CAVEAT_MW
    assert r.meta["metric"] == "NPAvgDD" and r.meta["fitness"] == "NP/Avg DD" and r.meta["n_iter"] == 36 and r.meta["shape"] == [6, 6]
    assert r.meta["n_windows"] == 1 and r.meta["n_complete"] == 1 and r.meta["strategy"] == "Synthetic_MW"
    assert r.wfc_np is not None and r.wfc_np.metric == "NP"
    assert all(c.passed for c in r.checks if c.severity == "error")
    assert len(r.plateau.windows) == 1 and len(r.selection.windows) == 1
    assert r.region.n_complete == 1 and r.region.p_lift < 0.05 and r.region.region_oos_mean > 0
    assert r.meta["wfc_reading"] == "edge" and r.meta["wfc_scoreable"] is True
    assert r.meta["region_k"] == 7 and r.meta["q"] == 0.2 and r.meta["n_distinct_median"] == 36


def _drift(grid, per_day, split=800):
    """The same $ added to every combination on every out-of-sample day (from `split`): the lift,
    its SD and its null do not move (the null keeps each day's grid mean) - only the grid's and
    the region's average out-of-sample result do, which is the matrix's column test."""
    grid.daily_pnl[:, split:] += per_day
    return grid


def test_verdict_matrix_has_the_lift_as_the_row_test_and_the_oos_sign_as_the_column():
    """Note § 6 step 6: L significant and R positive OOS -> edge (pass); L significant and R
    negative -> consistent loser (fail); L not significant and the grid average positive ->
    plateau (gate not applied); L not significant and the grid negative -> noise (fail)."""
    cases = (("persistent", 0.0, "edge", "pass"), ("persistent", -1000.0, "loser", "fail"),
             ("noise", 1000.0, "plateau", "plateau"), ("noise", -1000.0, "noise", "fail"))
    lifts = {}
    for structure, per_day, reading, verdict in cases:
        grid, groups, _ = _inputs(structure)
        r = run_multiwalk_battery(_drift(grid, per_day), groups, n_null=199, n_boot=50, seed=0)
        assert (r.meta["wfc_reading"], r.verdicts["wfc"]) == (reading, verdict), (structure, per_day, r.region.p_lift)
        assert r.meta["wfc_scoreable"] is True and r.gates_passed == int(verdict == "pass") and r.gates_total == 1
        lifts.setdefault(structure, set()).add((round(r.region.lift, 9), r.region.p_lift))
    assert all(len(v) == 1 for v in lifts.values())                  # the drift never touched the row test


def test_region_reading_table():
    for p, region, grid_mean, scoreable, expected in (
            (0.01, 5.0, 1.0, True, "edge"), (0.01, -5.0, 1.0, True, "loser"), (0.30, 5.0, 1.0, True, "plateau"),
            (0.30, 5.0, -1.0, True, "noise"), (float("nan"), 5.0, -1.0, True, "noise"),
            (0.01, 5.0, 1.0, False, "plateau"), (0.30, -5.0, -1.0, False, "plateau")):
        assert region_reading(p, region, grid_mean, scoreable=scoreable) == expected, (p, region, grid_mean, scoreable)
    assert set(READINGS) == {"edge", "loser", "plateau", "noise"}


def test_battery_records_the_null_and_keeps_torus_behind_the_flag():
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=99, n_boot=20, seed=0)
    assert r.meta["null"] == "signflip" and r.meta["block"] == 21
    assert r.wfc.null == "signflip" and r.wfc_np.null == "signflip" and r.wfc.windows[0].null.size == 99
    assert r.region.block == 21 and r.region.n_null == 99 and r.region.windows[0].null_lift.size == 99
    t = run_multiwalk_battery(grid, groups, n_null=99, n_boot=20, seed=0, null="torus")
    assert t.meta["null"] == "torus" and t.wfc.null == "torus" and t.wfc.windows[0].null.size == 35
    with pytest.raises(ValueError):
        run_multiwalk_battery(grid, groups, n_null=9, n_boot=9, seed=0, null="bootstrap")


def test_validation_fails_on_mismatched_names_and_counts():
    grid, groups, _ = _inputs("persistent", n_days=400)
    groups[0].param_names = ["A", "C"]
    with pytest.raises(MultiWalkValidationFailed) as e:
        run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)
    assert any(c.name == "param_names_match" and not c.passed for c in e.value.checks)
    groups[0].param_names = ["A", "B"]; groups[0].n_iterations = 35
    with pytest.raises(MultiWalkValidationFailed):
        run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)


def test_pick_off_grid_is_error_and_unknown_fitness_falls_back_to_np():
    grid, groups, _ = _inputs("persistent", n_days=400)
    groups[0].windows[0].params = (99.0, 99.0)
    with pytest.raises(MultiWalkValidationFailed):
        run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)
    grid, groups, _ = _inputs("persistent", n_days=400)
    groups[0].fitness_abbr, groups[0].fitness_name = "Sharpe", "Sharpe Ratio"
    r = run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)
    assert r.meta["metric"] == "NP" and r.wfc_np is None
    assert any(c.name == "fitness_supported" and c.severity == "warn" and not c.passed for c in r.checks)


def test_json_export_is_finite_and_complete():
    grid, groups, _ = _inputs("persistent", n_days=400)
    r = run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)
    d = json.loads(mw_to_json(r))
    assert set(d) >= {"meta", "checks", "wfc", "region", "plateau", "selection", "verdicts", "gates_passed", "gates_total", "caveat"}
    assert d["wfc"]["windows"][0]["null"] is not None and "x" in d["wfc"]["windows"][0]
    rw = d["region"]["windows"][0]
    assert len(rw["null_lift"]) == 20 and len(rw["region"]) == 7 and len(rw["ridge_is"]) == 36 and "wfc_reading" in d["meta"]
    json.dumps(d, allow_nan=False)


def test_min_trades_only_holes_the_continuity_correlation_never_the_region():
    """No per-cell min-trade hole in the region layer (note § 6.7): the correlation, kept for
    continuity, goes insufficient; the verdict still comes from the lift."""
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=10, min_trades=10**6)
    assert r.wfc.reasons == ["wfc_insufficient"] and r.verdicts["wfc"] == "pass" and r.gates_passed == 1
    mt = next(c for c in r.checks if c.name == "min_trades")
    assert not mt.passed and "[36]" in mt.detail
    assert np.isnan(r.wfc.windows[0].x).all() and np.isnan(r.wfc.windows[0].y).all()
    assert np.isfinite(r.region.windows[0].x).all()


def test_mismatched_parameter_count_is_a_validation_error_not_a_crash():
    grid, groups, _ = _inputs("persistent", n_days=400)
    groups[0].param_names = ["A", "B", "C"]
    groups[0].windows[0].params = (1.0, 2.0, 3.0)
    with pytest.raises(MultiWalkValidationFailed) as e:
        run_multiwalk_battery(grid, groups, n_null=20, n_boot=10)
    names = {c.name for c in e.value.checks if not c.passed}
    assert {"param_names_match", "picks_on_grid"} <= names


def test_unknown_group_no_is_a_validation_error():
    grid, groups, _ = _inputs("persistent", n_days=400)
    with pytest.raises(MultiWalkValidationFailed) as e:
        run_multiwalk_battery(grid, groups, group_no=999, n_null=20, n_boot=10)
    assert e.value.checks[0].name == "group_found"


def test_custom_window_schemes_pick_the_best_in_sample_variant():
    grid, groups, _ = _inputs("persistent", n_days=900)
    mw = run_multiwalk_battery(grid, groups, n_null=50, n_boot=20, seed=0)
    assert mw.meta["window_scheme"] == "multiwalk" and mw.meta["pick_label"] == "MultiWalk's pick"
    for scheme, n_windows in (("single", 1), ("two", 2)):
        r = run_multiwalk_battery(grid, groups, n_null=50, n_boot=20, seed=0, window_scheme=scheme)
        assert r.meta["window_scheme"] == scheme and r.meta["pick_label"] == "best in-sample"
        assert r.meta["n_windows"] == n_windows == len(r.wfc.windows) and r.meta["n_complete"] == n_windows
        for w, mwin in zip(r.wfc.windows, r.meta["windows"]):
            assert w.pick_index == int(np.nanargmax(w.x))
            assert tuple(mwin["params"]) == tuple(grid.params[w.pick_index])
            assert mwin["is_trades_median"] > 0 and mwin["oos_trades_median"] > 0
    with pytest.raises(ValueError):
        run_multiwalk_battery(grid, groups, n_null=10, n_boot=10, window_scheme="weekly")

def _near_identical(grid, seed=0, scale=0.02):
    """Every combination = one common daily series plus tiny noise: N_eff ~ 1, nothing to rank."""
    rng = np.random.default_rng(seed)
    common = rng.normal(5.0, 100.0, size=grid.daily_pnl.shape[1])
    grid.daily_pnl = common[None, :] + rng.normal(0.0, 100.0 * scale, size=grid.daily_pnl.shape)
    return grid


def test_nearly_identical_variants_take_the_plateau_branch():
    """Structure gate (note § 6 step 2): below NEFF_MIN effective variants the region is not
    scored - the parameter choice is immaterial, whatever the lift reads."""
    grid, groups, _ = _inputs("noise")
    r = run_multiwalk_battery(_near_identical(grid), groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] < NEFF_MIN == 3.0
    assert r.meta["wfc_scoreable"] is False and r.meta["wfc_reading"] == "plateau"
    assert r.verdicts["wfc"] == "plateau" and r.gates_passed == 0


def test_too_few_distinct_oos_patterns_take_the_plateau_branch_even_with_a_real_edge():
    """Identical variants out of sample (a parameter that does not bind there): 5 distinct
    patterns < 2 x 7 - any top-set statistic would only measure cluster self-alignment."""
    grid, groups, _ = _inputs("persistent")
    grid.daily_pnl[:, 800:] = grid.daily_pnl[np.arange(36) % 5, 800:]
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] >= NEFF_MIN and r.meta["n_distinct_median"] == 5
    assert r.meta["wfc_scoreable"] is False and r.verdicts["wfc"] == "plateau"


def test_distinct_variants_are_scored():
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] >= NEFF_MIN and r.meta["wfc_scoreable"] is True and r.verdicts["wfc"] == "pass"


def test_a_decayed_surface_never_passes():
    grid, groups, _ = _inputs("decay")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.region.lift < 0 and r.region.p_lift >= 0.05 and r.verdicts["wfc"] != "pass"