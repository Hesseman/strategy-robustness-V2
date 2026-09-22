import json

import numpy as np
import pytest

from robustness.multiwalk_battery import CAVEAT_MW, MultiWalkValidationFailed, mw_to_json, run_multiwalk_battery
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
