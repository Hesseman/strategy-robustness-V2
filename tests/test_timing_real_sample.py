"""Timing-sensitivity oracle on the user's real MNQ 30-min sample. Skipped when the files
are absent. Never copy these files into the repo."""
import os
from pathlib import Path

import numpy as np
import pytest

from robustness.bars_loader import load_bars
from robustness.join import join_trades_to_bars
from robustness.report_parser import parse_report
from robustness.returns import usd_per_contract
from robustness.timing import run_timing

REPORT = Path(os.environ.get("SR_SAMPLE_REPORT", "C:/Users/User/Downloads/MNQ30MDSP.csv"))
BARS = Path(os.environ.get("SR_SAMPLE_BARS", "C:/Users/User/Downloads/MNQ30M_BarData_Tradestation.txt"))

pytestmark = pytest.mark.skipif(not (REPORT.exists() and BARS.exists()), reason="real sample files not present")


def test_real_sample_timing_curves():
    rep = parse_report(REPORT.read_text(encoding="utf-8-sig"))
    bars = load_bars(BARS.read_text(encoding="utf-8-sig"))
    r = run_timing(rep, bars, max_k=10)
    joined = join_trades_to_bars(rep.trades, bars, summary=rep.summary, settings=rep.settings)
    assert r.meta["n_trades"] == 418
    assert r.entry.points[0].n_alive == 418 and r.exit.points[0].n_alive == 418
    assert r.entry.points[0].total_usd == pytest.approx(usd_per_contract(joined.trades, joined.point_value).sum())
    for c in (r.entry, r.exit, r.entry_early, r.exit_early):
        assert all(np.isfinite([p.total_usd, p.mean_usd, p.mean_pct, p.win_rate]).all() for p in c.points)
        print(f"\n{c.kind.upper()} {c.shift.upper()} ({c.mode}), at-open share {r.at_open_share:.0%}, first non-positive k = {c.first_nonpositive_k}")
        for p in c.points:
            print(f"  k={p.k:2d} alive={p.n_alive:3d} total=${p.total_usd:,.0f} mean%={p.mean_pct*100:+.3f} "
                  f"win={p.win_rate:.0%} pf={p.profit_factor:.2f} retention={p.retention:.2f}")
