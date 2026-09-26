"""Orchestrates the MultiWalk surface section: text + db -> validation checks -> windows -> metrics
-> WFC (gate: the region lift of region_wfc, with Tinsley's correlation kept as the continuity
number), plateau (score), selection haircut (reference) -> verdicts and JSON (spec decisions 3-7,
12). Only reason to change: the section's composition or verdict rules."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from robustness.battery import _jsonable
from robustness.join import Check
from robustness.multiwalk_text import MultiWalkGrid
from robustness.plateau_grid import PlateauResult, plateau_test
from robustness.region_wfc import Q_TOP, RegionResult, region_test
from robustness.selection import SelectionResult, selection_test
from robustness.surface import metric_values, window_metrics
from robustness.walkforward_db import WFGroup
from robustness.wfc_grid import NULLS, WFCResult, wfc_test
from robustness.windows import SCHEMES, Window, custom_windows, derive_windows

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
    region: RegionResult
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
                        + ("" if grid.is_full_grid else " - not a full product grid (the plateau neighbourhoods are sparser; the torus null, if chosen, permutes)"), severity="warn"))
    bad = []
    for w in group.windows:
        if not (1 <= w.grid_row <= grid.n_iter) or len(w.params) != grid.params.shape[1] or not np.allclose(grid.params[w.grid_row - 1], w.params, atol=1e-9):
            bad.append(f"row {w.grid_row} -> {w.params}")
    checks.append(Check("picks_on_grid", not bad, "every window's pick matches its grid row" if not bad else "mismatch: " + "; ".join(bad), severity="error"))
    checks.append(Check("windows_present", len(group.windows) > 0, f"{len(group.windows)} walk-forward windows", severity="error"))
    ok = group.fitness_abbr in _SUPPORTED_FITNESS
    checks.append(Check("fitness_supported", ok, f"fitness {group.fitness_abbr} ({group.fitness_name})"
                        + ("" if ok else " is not reproduced by this app - using Net Profit"), severity="warn"))
    return checks


# Below this many effective independent variants (the selection card's participation ratio) the
# combinations are nearly one strategy, and a significant lift rests on the handful of trades where
# they differ. A caution printed beside a PASS (meta wfc_guards.few_variants), never a gate: the
# sign-flip null holds however alike the variants are, and real grids sit at 1.0-1.7, so a gate here
# hid every significant ridge but one (decision 2026-09-25, docs/wfc-region-lift.md).
NEFF_MIN = 3.0
# The verdict matrix of docs/wfc-region-lift.md ('The verdict'): Tinsley's, with the lift as the row test.
READINGS = {"edge": "structural edge, localised in the in-sample top region",
            "loser": "consistent loser - the top region beats the grid average but loses money",
            "plateau": "plateau - parameter choice immaterial, the strategy-level tests govern",
            "noise": "noise - no edge"}
_GATE = {"edge": "pass", "loser": "fail", "noise": "fail", "plateau": "plateau"}


def region_reading(p_lift: float, region_oos_mean: float, grid_oos_mean: float, *, scoreable: bool,
                   alpha: float = 0.05) -> str:
    """The verdict matrix with the region lift as the row test.

    Accepts: the lift's p, the region's and the grid's mean out-of-sample net profit, whether the
    grid can be scored (distinct OOS daily patterns >= 2 x region size) and alpha.
    Returns a READINGS key: 'plateau' when not scoreable; otherwise a significant lift reads
    'edge' (region positive out of sample) or 'loser', a non-significant one 'plateau' (grid
    average positive out of sample) or 'noise'. Guarantees a NaN p counts as not significant."""
    if not scoreable:
        return "plateau"
    if np.isfinite(p_lift) and p_lift < alpha:
        return "edge" if region_oos_mean > 0 else "loser"
    return "plateau" if grid_oos_mean > 0 else "noise"


def nulls_disagree(p_flip: float, p_boot: float, alpha: float = 0.05) -> bool:
    """The cross-check's rule, fixed a priori (docs/wfc-region-lift.md, 'Decisions'): True when the
    lift's sign-flip p and its centred block bootstrap p fall on different sides of alpha - one
    < alpha, the other >= alpha. A NaN p counts as not significant, as in region_reading."""
    return bool((np.isfinite(p_flip) and p_flip < alpha) != (np.isfinite(p_boot) and p_boot < alpha))


def _scoreable(n_distinct: float, k: int) -> bool:
    """False when the out-of-sample results repeat - fewer distinct daily patterns than twice the
    region, so a top-k set would only measure clusters of identical variants lining up with
    themselves; a NaN count never blocks. N_eff deliberately plays no part (see NEFF_MIN)."""
    return not (np.isfinite(n_distinct) and n_distinct < 2 * k)


def run_multiwalk_battery(grid: MultiWalkGrid, groups: list[WFGroup], *, group_no: int | None = None, n_null: int = 999,
                          n_boot: int = 500, seed: int = 0, min_trades: int = 10, alpha: float = 0.05,
                          window_scheme: str = "multiwalk", null: str = "signflip", block: int = 21) -> MultiWalkResult:
    """Run the three surface tests for one walk-forward group.

    Accepts: the parsed grid and groups; group_no selects a group (default the first; an unknown number raises MultiWalkValidationFailed); n_null WFC
    null draws; n_boot Reality-Check draws; seed; min_trades - iterations with fewer closed
    trades in a window's IS or OOS are dropped from that window; alpha - WFC gate level;
    window_scheme - 'multiwalk' (the DB schedule and MultiWalk's picks), 'two' or 'single'
    (windows.custom_windows; each window's pick is then the best in-sample variant by the metric);
    null - the null of Tinsley's correlation, 'signflip' (block sign-flip of the OOS daily
    cross-sectional deviations, block trading days per block) or 'torus' (the pre-2026-09-25 grid
    shift, comparison only); the region lift always uses the sign-flip with the same block.
    Returns: MultiWalkResult with meta (incl. window_scheme, pick_label, null, block, per window the
    median closed trades per variant in and out of sample and the window's reading), checks,
    windows, the WFC correlation on the project's fitness (and on Net Profit when the fitness is
    NP/AvgDD) - the continuity number, never the gate - the region result (region_wfc.region_test
    on net profit, n_null draws), plateau, selection and verdicts {'wfc': pass|fail|plateau|
    insufficient, 'plateau': 'score', 'selection': 'reference'}. The WFC verdict is the region
    matrix (region_reading, meta wfc_reading): pass = edge; fail = loser or noise; plateau = the
    lift is not significant on a grid positive out of sample, or the grid cannot be scored (median
    distinct OOS patterns over the complete windows < 2 x the region size; meta wfc_scoreable,
    n_distinct_median); insufficient = no complete window. gates 0/1 of 1. meta wfc_guards - the
    facts printed beside a PASS - holds n_eff_median and few_variants (< NEFF_MIN; context, never
    a gate), the region's and the grid's mean OOS net profit, per complete window the region's
    margin over the grid (region_minus_grid, windows_ahead of windows_complete), and the lift's p
    under the centred block bootstrap (p_lift_boot; the cross-check, never the gate) with
    nulls_disagree; each meta window carries its own region_minus_grid.
    Guarantees: raises ValueError for an unknown window_scheme or null and MultiWalkValidationFailed
    when an error-severity check fails; nothing downstream runs then; min_trades never reaches the
    region verdict; deterministic for a given seed."""
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    if window_scheme not in SCHEMES:
        raise ValueError(f"window_scheme must be one of {SCHEMES}, got {window_scheme!r}")
    if group_no is None:
        group = groups[0]
    else:
        group = next((g for g in groups if g.group_no == group_no), None)
        if group is None:
            raise MultiWalkValidationFailed([Check("group_found", False, f"walk-forward group {group_no} is not in the database (groups: {[g.group_no for g in groups]})", severity="error")])
    checks = validate(grid, group)
    if any(not c.passed and c.severity == "error" for c in checks):
        raise MultiWalkValidationFailed(checks)
    windows = derive_windows(group, grid.dates) if window_scheme == "multiwalk" else custom_windows(grid.dates, window_scheme)
    metric = _SUPPORTED_FITNESS.get(group.fitness_abbr, "NP")
    pairs, pairs_np, dropped, trades = [], [], [], []
    for w in windows:
        mi, mo = window_metrics(grid, w.is_mask), window_metrics(grid, w.oos_mask)
        keep = (mi.n_trades >= min_trades) & (mo.n_trades >= min_trades)
        dropped.append(int((~keep).sum()))
        trades.append((int(np.median(mi.n_trades)), int(np.median(mo.n_trades))))
        x, y = metric_values(mi, metric).copy(), metric_values(mo, metric).copy()
        x[~keep] = np.nan; y[~keep] = np.nan
        if window_scheme != "multiwalk":   # MultiWalk picked nothing for these windows: take the best in-sample variant
            pick = int(np.nanargmax(x)) if np.isfinite(x).any() else 0
            w.grid_row, w.params = pick + 1, tuple(float(v) for v in grid.params[pick])
        pairs.append((x, y))
        xn, yn = mi.net_profit.copy(), mo.net_profit.copy()
        xn[~keep] = np.nan; yn[~keep] = np.nan
        pairs_np.append((xn, yn))
    checks.append(Check("min_trades", all(d == 0 for d in dropped),
                        f"iterations dropped for fewer than {min_trades} trades in IS or OOS, per window: {dropped}", severity="warn"))
    wfc = wfc_test(pairs, windows, grid, metric=metric, alpha=alpha, n_null=n_null, seed=seed, null=null, block=block)
    wfc_np = (wfc_test(pairs_np, windows, grid, metric="NP", alpha=alpha, n_null=n_null, seed=seed, null=null, block=block)
              if metric != "NP" else None)
    region = region_test(grid, windows, n_null=n_null, seed=seed, block=block)
    plateau = plateau_test(pairs, windows, grid)
    selection = selection_test(grid, windows, n_boot=n_boot, seed=seed)
    complete = [i for i, w in enumerate(windows) if w.complete]
    neffs = [selection.windows[i].n_eff for i in complete if np.isfinite(selection.windows[i].n_eff)]
    n_eff_median = float(np.median(neffs)) if neffs else float("nan")
    n_distinct_median = float(np.median([region.windows[i].n_distinct_oos for i in complete])) if complete else float("nan")
    scoreable = _scoreable(n_distinct_median, region.k)
    reading = (region_reading(region.p_lift, region.region_oos_mean, region.grid_oos_mean, scoreable=scoreable, alpha=alpha)
               if region.n_complete else "insufficient")
    wfc_verdict = _GATE.get(reading, "insufficient")
    verdicts = {"wfc": wfc_verdict, "plateau": "score", "selection": "reference"}
    window_readings = [region_reading(rw.p_lift, rw.region_oos_mean, rw.grid_oos_mean, alpha=alpha,
                                      scoreable=_scoreable(rw.n_distinct_oos, rw.k)) for rw in region.windows]
    margins = [rw.region_oos_mean - rw.grid_oos_mean for rw in region.windows]
    guards = {"n_eff_median": n_eff_median, "few_variants": bool(np.isfinite(n_eff_median) and n_eff_median < NEFF_MIN),
              "region_oos_mean": region.region_oos_mean, "grid_oos_mean": region.grid_oos_mean,
              "region_minus_grid": [margins[i] for i in complete], "windows_ahead": sum(margins[i] > 0 for i in complete),
              "windows_complete": len(complete), "p_lift_boot": region.p_lift_boot,
              "nulls_disagree": nulls_disagree(region.p_lift, region.p_lift_boot, alpha)}
    oos_trades = [trades[i][1] for i in complete]
    meta = {"strategy": group.strategy, "symbol": group.symbol, "interval": group.interval, "group": group.label,
            "fitness": group.fitness_name, "fitness_abbr": group.fitness_abbr, "metric": metric,
            "param_names": grid.param_names, "shape": list(grid.shape), "n_iter": grid.n_iter,
            "dates_start": grid.dates[0], "dates_end": grid.dates[-1], "n_days": grid.n_days,
            "n_windows": len(windows), "n_complete": sum(w.complete for w in windows),
            "n_eff_median": n_eff_median, "n_distinct_median": n_distinct_median, "wfc_scoreable": bool(scoreable),
            "wfc_reading": reading, "wfc_guards": guards, "region_k": region.k, "q": Q_TOP,
            "oos_trades_median": float(np.median(oos_trades)) if oos_trades else float("nan"),
            "window_scheme": window_scheme, "pick_label": "MultiWalk's pick" if window_scheme == "multiwalk" else "best in-sample",
            "windows": [{"index": w.index, "label": w.label, "is_start": w.is_start, "is_end": w.is_end, "oos_start": w.oos_start,
                         "oos_end": w.oos_end, "complete": w.complete, "grid_row": w.grid_row, "params": list(w.params),
                         "is_trades_median": tr[0], "oos_trades_median": tr[1], "wfc_reading": rd, "region_minus_grid": mg}
                        for w, tr, rd, mg in zip(windows, trades, window_readings, margins)],
            "n_null": n_null, "n_boot": n_boot, "seed": seed, "min_trades": min_trades, "alpha": alpha, "null": null, "block": block,
            "in_period": f"{group.in_len} {group.in_type}", "out_period": f"{group.out_len} {group.out_type}", "anchored": group.anchored}
    return MultiWalkResult(meta=meta, checks=checks, windows=windows, wfc=wfc, wfc_np=wfc_np, region=region, plateau=plateau,
                           selection=selection, verdicts=verdicts, gates_passed=int(wfc_verdict == "pass"), gates_total=1,
                           caveat=CAVEAT_MW)


def mw_to_json(result: MultiWalkResult) -> str:
    """JSON of the whole result (numpy/pandas/dataclasses converted; NaN/inf -> null; the window
    masks are dropped as they are derivable from the dates)."""
    d = _jsonable(result)
    for w in d.get("windows", []):
        w.pop("is_mask", None); w.pop("oos_mask", None)
    return json.dumps(d, indent=2, default=str)
