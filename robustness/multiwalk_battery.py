"""Orchestrates the MultiWalk surface section: text + db -> validation checks -> windows -> metrics
-> WFC (gate), plateau (score), selection haircut (reference) -> verdicts and JSON (spec decisions
3-7, 12). Only reason to change: the section's composition or verdict rules."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from robustness.battery import _jsonable
from robustness.join import Check
from robustness.multiwalk_text import MultiWalkGrid
from robustness.plateau_grid import PlateauResult, plateau_test
from robustness.selection import SelectionResult, selection_test
from robustness.surface import metric_values, window_metrics
from robustness.walkforward_db import WFGroup
from robustness.wfc_grid import WFCResult, wfc_test
from robustness.windows import Window, derive_windows

CAVEAT_MW = ("These three tests read every parameter combination this MultiWalk optimisation tried, so unlike "
             "the cards above they do see how many variants were compared - but only inside this one project. "
             "They know nothing about other strategies, symbols or grids that were tried before it.")
_SUPPORTED_FITNESS = {"NPAvgDD": "NPAvgDD", "NP": "NP"}


class MultiWalkValidationFailed(Exception):
    """The text file and the database do not describe the same optimisation; .checks carries the diagnosis."""

    def __init__(self, checks: list[Check]):
        super().__init__("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed and c.severity == "error"))
        self.checks = checks


@dataclass
class MultiWalkResult:
    meta: dict
    checks: list[Check]
    windows: list[Window]
    wfc: WFCResult
    wfc_np: WFCResult | None
    plateau: PlateauResult
    selection: SelectionResult
    verdicts: dict
    gates_passed: int
    gates_total: int
    caveat: str


def validate(grid: MultiWalkGrid, group: WFGroup) -> list[Check]:
    """Cross-check the text grid against the database group. Returns the checks (error severity
    blocks the run; warn severity is informational)."""
    checks: list[Check] = []
    same = grid.param_names == group.param_names
    checks.append(Check("param_names_match", same, f"text file inputs {grid.param_names} vs database {group.param_names}", severity="error"))
    checks.append(Check("iteration_count", grid.n_iter == group.n_iterations,
                        f"{grid.n_iter} iterations in the text file, {group.n_iterations} in the database", severity="error"))
    checks.append(Check("full_grid", grid.is_full_grid, f"grid {grid.shape} = {int(np.prod(grid.shape))} cells for {grid.n_iter} iterations"
                        + ("" if grid.is_full_grid else " - not a full product grid, the WFC null uses permutations"), severity="warn"))
    bad = []
    for w in group.windows:
        if not (1 <= w.grid_row <= grid.n_iter) or not np.allclose(grid.params[w.grid_row - 1], w.params, atol=1e-9):
            bad.append(f"row {w.grid_row} -> {w.params}")
    checks.append(Check("picks_on_grid", not bad, "every window's pick matches its grid row" if not bad else "mismatch: " + "; ".join(bad), severity="error"))
    checks.append(Check("windows_present", len(group.windows) > 0, f"{len(group.windows)} walk-forward windows", severity="error"))
    ok = group.fitness_abbr in _SUPPORTED_FITNESS
    checks.append(Check("fitness_supported", ok, f"fitness {group.fitness_abbr} ({group.fitness_name})"
                        + ("" if ok else " is not reproduced by this app - using Net Profit"), severity="warn"))
    return checks


def run_multiwalk_battery(grid: MultiWalkGrid, groups: list[WFGroup], *, group_no: int | None = None, n_null: int = 999,
                          n_boot: int = 500, seed: int = 0, min_trades: int = 10, alpha: float = 0.05) -> MultiWalkResult:
    """Run the three surface tests for one walk-forward group.

    Accepts: the parsed grid and groups; group_no selects a group (default the first); n_null WFC
    null draws; n_boot Reality-Check draws; seed; min_trades - iterations with fewer closed
    trades in a window's IS or OOS are dropped from that window; alpha - WFC gate level.
    Returns: MultiWalkResult with meta, checks, windows, the WFC result on the project's fitness
    (and on Net Profit when the fitness is NP/AvgDD), plateau, selection, verdicts
    {'wfc': pass|fail|insufficient, 'plateau': 'score', 'selection': 'reference'}, gates 0/1 of 1.
    Guarantees: raises MultiWalkValidationFailed when an error-severity check fails; nothing
    downstream runs then; deterministic for a given seed."""
    group = next((g for g in groups if g.group_no == group_no), groups[0]) if group_no is not None else groups[0]
    checks = validate(grid, group)
    if any(not c.passed and c.severity == "error" for c in checks):
        raise MultiWalkValidationFailed(checks)
    windows = derive_windows(group, grid.dates)
    metric = _SUPPORTED_FITNESS.get(group.fitness_abbr, "NP")
    pairs, pairs_np, dropped = [], [], []
    for w in windows:
        mi, mo = window_metrics(grid, w.is_mask), window_metrics(grid, w.oos_mask)
        keep = (mi.n_trades >= min_trades) & (mo.n_trades >= min_trades)
        dropped.append(int((~keep).sum()))
        x, y = metric_values(mi, metric).copy(), metric_values(mo, metric).copy()
        x[~keep] = np.nan; y[~keep] = np.nan
        pairs.append((x, y))
        xn, yn = mi.net_profit.copy(), mo.net_profit.copy()
        xn[~keep] = np.nan; yn[~keep] = np.nan
        pairs_np.append((xn, yn))
    checks.append(Check("min_trades", all(d == 0 for d in dropped),
                        f"iterations dropped for fewer than {min_trades} trades in IS or OOS, per window: {dropped}", severity="warn"))
    wfc = wfc_test(pairs, windows, grid, metric=metric, alpha=alpha, n_null=n_null, seed=seed)
    wfc_np = wfc_test(pairs_np, windows, grid, metric="NP", alpha=alpha, n_null=n_null, seed=seed) if metric != "NP" else None
    plateau = plateau_test(pairs, windows, grid)
    selection = selection_test(grid, windows, n_boot=n_boot, seed=seed)
    verdicts = {"wfc": "pass" if wfc.passed else ("insufficient" if "wfc_insufficient" in wfc.reasons else "fail"),
                "plateau": "score", "selection": "reference"}
    meta = {"strategy": group.strategy, "symbol": group.symbol, "interval": group.interval, "group": group.label,
            "fitness": group.fitness_name, "fitness_abbr": group.fitness_abbr, "metric": metric,
            "param_names": grid.param_names, "shape": list(grid.shape), "n_iter": grid.n_iter,
            "dates_start": grid.dates[0], "dates_end": grid.dates[-1], "n_days": grid.n_days,
            "n_windows": len(windows), "n_complete": sum(w.complete for w in windows),
            "windows": [{"index": w.index, "label": w.label, "is_start": w.is_start, "is_end": w.is_end, "oos_start": w.oos_start,
                         "oos_end": w.oos_end, "complete": w.complete, "grid_row": w.grid_row, "params": list(w.params)} for w in windows],
            "n_null": n_null, "n_boot": n_boot, "seed": seed, "min_trades": min_trades, "alpha": alpha,
            "in_period": f"{group.in_len} {group.in_type}", "out_period": f"{group.out_len} {group.out_type}", "anchored": group.anchored}
    return MultiWalkResult(meta=meta, checks=checks, windows=windows, wfc=wfc, wfc_np=wfc_np, plateau=plateau, selection=selection,
                           verdicts=verdicts, gates_passed=int(wfc.passed), gates_total=1, caveat=CAVEAT_MW)


def mw_to_json(result: MultiWalkResult) -> str:
    """JSON of the whole result (numpy/pandas/dataclasses converted; NaN/inf -> null; the window
    masks are dropped as they are derivable from the dates)."""
    d = _jsonable(result)
    for w in d.get("windows", []):
        w.pop("is_mask", None); w.pop("oos_mask", None)
    return json.dumps(d, indent=2, default=str)
