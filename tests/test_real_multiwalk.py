"""Opt-in oracle on the user's real LE2601 MultiWalk project (never committed). Set
SR_SAMPLE_MW_DIR to a MultiWalk project folder that was run with
iUseLegacyOptimizationTextFileFormat: true."""
import csv
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.surface import daily_drawdown_stats
from robustness.walkforward_db import parse_walkforward_db

MW_DIR = Path(os.environ.get("SR_SAMPLE_MW_DIR", "E:/Portfolio Strategies/2 - Bank/LE2601_DSP_30M_LS_MW"))
TXT = next(iter(glob.glob(str(MW_DIR / "Optimization Files" / "*_MultiWalk.txt"))), None)
DB = MW_DIR / "Walkforward Files" / "WalkforwardData.db"
TRADES = next(iter(glob.glob(str(MW_DIR / "Walkforward Files" / "*TradeData.csv"))), None)
EQUITY = next(iter(glob.glob(str(MW_DIR / "Walkforward Files" / "*EquityData.csv"))), None)
DETAILS = MW_DIR / "Walkforward Files" / "Walkforward In-Out Periods Analysis Details.csv"
pytestmark = pytest.mark.skipif(not (TXT and DB.exists() and TRADES and EQUITY and DETAILS.exists()),
                                reason="real MultiWalk sample not available (SR_SAMPLE_MW_DIR)")


@pytest.fixture(scope="module")
def grid():
    return parse_multiwalk_text(Path(TXT).read_text(encoding="utf-8", errors="replace"))


@pytest.fixture(scope="module")
def groups():
    return parse_walkforward_db(DB.read_bytes())


def _ymd(mdy: str) -> str:
    m, d, y = mdy.split("/")
    return f"{y}{m}{d}"


def test_grid_shape_names_and_picks_line_up_with_the_db(grid, groups):
    g = groups[0]
    assert grid.param_names == g.param_names and grid.n_iter == g.n_iterations and grid.is_full_grid
    for w in g.windows:
        assert tuple(grid.params[w.grid_row - 1]) == pytest.approx(w.params)
    assert grid.dates[0] <= pd.Timestamp("2015-02-01") and grid.dates[-1] >= pd.Timestamp("2026-06-12")


def test_pick_closed_pnl_reconciles_with_the_walkforward_trade_file(grid, groups):
    rows = list(csv.DictReader(open(TRADES, encoding="utf-8-sig")))
    for k, w in enumerate(groups[0].windows[:2]):
        lo, hi = w.oos_start.strftime("%Y%m%d"), w.oos_end.strftime("%Y%m%d")
        wf = sum(float(r["PNL"]) for r in rows if r["Type"] == "Exit" and lo <= _ymd(r["Date"]) <= hi)
        i = w.grid_row - 1
        d = grid.exit_dates[i].astype("datetime64[D]")
        sel = (d >= np.datetime64(w.oos_start.date())) & (d <= np.datetime64(w.oos_end.date()))
        txt = float(grid.exit_pnl[i][sel].sum())
        tol = 200.0 if k == 0 else 0.01   # window 1: the walk-forward restarts at the window start (one boundary trade)
        assert abs(txt - wf) <= tol, (k, txt, wf)


def test_drawdown_definitions_reproduce_multiwalks_report():
    det = next(csv.DictReader(open(DETAILS, encoding="utf-8-sig")))
    lo, hi = _ymd(det["IS Begin Date"]), _ymd(det["IS End Date"])
    eq = list(csv.DictReader(open(EQUITY, encoding="utf-8-sig")))
    first = min(_ymd(r["Date"]) for r in eq)
    pnl = np.array([float(r["Daily M2M Equity"]) for r in eq if max(lo, first) <= _ymd(r["Date"]) <= hi])
    net, max_dd, avg_dd = daily_drawdown_stats(pnl)
    assert abs(net - float(det["IS Net Profit"])) < 1.0
    assert abs(max_dd - float(det["IS Max DD"])) < 1.0
    assert abs(avg_dd - float(det["IS Avg DD"])) < 0.02 * abs(float(det["IS Avg DD"]))


def test_real_battery_runs_and_prints(grid, groups):
    from robustness.multiwalk_battery import run_multiwalk_battery
    r = run_multiwalk_battery(grid, groups, n_null=999, n_boot=200, seed=0)
    assert r.meta["n_complete"] == 2 and r.verdicts["wfc"] in ("pass", "fail", "plateau")
    assert r.meta["wfc_scoreable"] or r.verdicts["wfc"] == "plateau"
    w = r.wfc.windows
    print("REAL MULTIWALK:", r.verdicts, r.meta["wfc_reading"], "lift=", round(r.region.lift, 3), "p=", r.region.p_lift,
          "| pooled rho=", round(r.wfc.pooled_spearman, 3), "p=", r.wfc.pooled_p,
          "posOOS=", round(r.wfc.pooled_pos_oos_frac, 3), "| per window rho=", [round(x.spearman, 3) for x in w],
          "p=", [x.p_value for x in w], "| plateau=", [round(p.score_oos, 2) for p in r.plateau.windows],
          "| p_pick=", [s.p_pick for s in r.selection.windows],
          "n_eff=", [round(s.n_eff, 1) for s in r.selection.windows], "n_eff_median=", round(r.meta["n_eff_median"], 2))


def test_region_layer_reproduces_the_notes_le2601_row_and_reads_plateau(grid, groups):
    """Oracle from docs/research/2026-09-25-wfc-region-concordance.md § 3 (LE2601, net profit,
    sign-flip null, 499 draws, seed 0 - the prototype's settings): MultiWalk's windows give a
    top-20% lift of +0.32 SD at p = 0.156, overlap precision 0.29 (p 0.30), ridge Jaccard 0.21 and
    a 2.9-step ridge shift; the 'two' scheme +0.32 at p = 0.17, precision 0.44 (p 0.092). The
    production module draws exactly as the prototype did, so the p-values reproduce to the draw.
    Through the battery (999 draws) the lift stays insignificant and N_eff ~1.5 takes the plateau
    branch anyway."""
    from robustness.multiwalk_battery import run_multiwalk_battery
    from robustness.region_wfc import region_test
    from robustness.windows import custom_windows, derive_windows
    mw = region_test(grid, derive_windows(groups[0], grid.dates), n_null=499, seed=0)
    assert mw.n_complete == 2 and mw.k == 48
    assert mw.lift == pytest.approx(0.32, abs=0.005) and mw.p_lift == pytest.approx(0.156)
    assert mw.precision == pytest.approx(0.29, abs=0.005) and mw.p_precision == pytest.approx(0.302)
    assert mw.ridge_jaccard == pytest.approx(0.21, abs=0.005) and mw.ridge_shift == pytest.approx(2.9, abs=0.05)
    two_windows = custom_windows(grid.dates, "two")
    two = region_test(grid, two_windows, n_null=499, seed=0)
    assert two.lift == pytest.approx(0.32, abs=0.005) and two.p_lift == pytest.approx(0.172)
    assert two.precision == pytest.approx(0.44, abs=0.005) and two.p_precision == pytest.approx(0.092)
    r = run_multiwalk_battery(grid, groups, n_null=999, n_boot=50, seed=0)
    assert r.region.lift == pytest.approx(mw.lift) and 0.10 <= r.region.p_lift <= 0.25
    assert r.meta["n_eff_median"] < 3.0 and r.meta["wfc_scoreable"] is False
    assert r.meta["wfc_reading"] == "plateau" and r.verdicts["wfc"] == "plateau" and r.gates_passed == 0
    print("LE2601 region lift: MultiWalk windows", round(mw.lift, 3), "p", mw.p_lift, "precision", round(mw.precision, 3),
          "| two", round(two.lift, 3), "p", two.p_lift, "| battery p (999)", r.region.p_lift, r.meta["wfc_reading"])


def test_signflip_null_takes_le2601_off_the_torus_pass(grid, groups):
    """Oracle from docs/research/2026-09-25-wfc-region-concordance.md: on the 'two' scheme the
    torus null passed LE2601's NP/AvgDD surface (p = 0.022) and the sign-flip null does not
    (p = 0.165); on the MultiWalk schedule net profit moves from p = 0.028 to 0.066. The bands
    below leave room for Monte-Carlo noise at 999 draws."""
    from robustness.multiwalk_battery import run_multiwalk_battery
    torus = run_multiwalk_battery(grid, groups, n_null=999, n_boot=50, seed=0, window_scheme="two", null="torus")
    flip = run_multiwalk_battery(grid, groups, n_null=999, n_boot=50, seed=0, window_scheme="two")
    assert torus.wfc.pooled_spearman == pytest.approx(flip.wfc.pooled_spearman)     # the statistic is untouched, only its null
    assert torus.wfc.pooled_p < 0.05 and flip.wfc.pooled_p > 0.10
    assert flip.meta["null"] == "signflip" and flip.wfc.null == "signflip" and torus.wfc.null == "torus"
    mw = run_multiwalk_battery(grid, groups, n_null=999, n_boot=50, seed=0)
    assert 0.04 <= mw.wfc_np.pooled_p <= 0.12
    print("LE2601 two-scheme NP/AvgDD pooled p: torus", torus.wfc.pooled_p, "sign-flip", flip.wfc.pooled_p,
          "| MultiWalk schedule NP pooled p sign-flip", mw.wfc_np.pooled_p)
