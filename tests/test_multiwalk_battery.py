import json
from dataclasses import replace

import numpy as np
import pytest

from robustness.multiwalk_battery import CAVEAT_MW, NEFF_MIN, MultiWalkValidationFailed, mw_to_json, run_multiwalk_battery, wfc_quadrant
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


def test_noise_fails_wfc_gate():
    grid, groups, _ = _inputs("noise")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.verdicts["wfc"] == "fail" and r.gates_passed == 0


def test_battery_records_the_null_and_keeps_torus_behind_the_flag():
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=99, n_boot=20, seed=0)
    assert r.meta["null"] == "signflip" and r.meta["block"] == 21
    assert r.wfc.null == "signflip" and r.wfc_np.null == "signflip" and r.wfc.windows[0].null.size == 99
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
    assert set(d) >= {"meta", "checks", "wfc", "plateau", "selection", "verdicts", "gates_passed", "gates_total", "caveat"}
    assert d["wfc"]["windows"][0]["null"] is not None and "x" in d["wfc"]["windows"][0]


def test_min_trades_drop_path_and_insufficient_verdict():
    grid, groups, _ = _inputs("persistent", n_days=400)
    r = run_multiwalk_battery(grid, groups, n_null=20, n_boot=10, min_trades=10**6)
    assert r.verdicts["wfc"] == "insufficient" and r.gates_passed == 0
    mt = next(c for c in r.checks if c.name == "min_trades")
    assert not mt.passed and "[36]" in mt.detail
    assert np.isnan(r.wfc.windows[0].x).all() and np.isnan(r.wfc.windows[0].y).all()


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


def test_a_fail_on_nearly_identical_variants_reads_not_informative():
    grid, groups, _ = _inputs("noise")
    r = run_multiwalk_battery(_near_identical(grid), groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] < NEFF_MIN == 3.0
    assert not r.wfc.passed and r.verdicts["wfc"] == "not_informative" and r.gates_passed == 0


def test_distinct_variants_keep_their_verdicts():
    grid, groups, _ = _inputs("persistent")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] >= NEFF_MIN and r.verdicts["wfc"] == "pass"
    grid, groups, _ = _inputs("noise")
    r = run_multiwalk_battery(grid, groups, n_null=199, n_boot=50, seed=0)
    assert r.meta["n_eff_median"] >= NEFF_MIN and r.verdicts["wfc"] == "fail"


def test_wfc_quadrant_relabels_only_low_correlation_windows_with_few_effective_variants():
    grid, groups, _ = _inputs("persistent", n_days=400)
    w = run_multiwalk_battery(grid, groups, n_null=20, n_boot=10).wfc.windows[0]
    for q in ("spurious result, high over-fitting", "noise, no edge"):
        assert wfc_quadrant(replace(w, quadrant=q), 1.5) == "not informative - variants nearly identical"
        assert wfc_quadrant(replace(w, quadrant=q), 5.0) == q and wfc_quadrant(replace(w, quadrant=q), float("nan")) == q
    for q in ("structural edge, low over-fitting", "consistently loss-making strategy"):
        assert wfc_quadrant(replace(w, quadrant=q), 1.5) == q