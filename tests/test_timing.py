import json

import numpy as np
import pandas as pd
import pytest

from robustness.bars_loader import load_bars
from robustness.battery import ValidationFailed, _jsonable
from robustness.report_parser import parse_report
from robustness.returns import pct_returns, usd_per_contract
from robustness.timing import TimingResult, delay_curve, delayed_returns, run_timing, to_json
from conftest import make_bars, make_trades, bars_to_ts_text, trades_to_report_text

PV = 2.0


def _ramp(n=40):
    """open = 100 + i on bar i, so every delayed fill price is exact."""
    idx = pd.date_range("2025-01-02 08:30", periods=n, freq="30min", name="ts")
    o = 100.0 + np.arange(n)
    return pd.DataFrame({"open": o, "high": o + 1.0, "low": o - 1.0, "close": o + 0.5, "volume": 100.0}, index=idx)


def _trade(entry_idx, exit_idx, direction=1, entry_price=None, exit_price=None, bars=None):
    bars = _ramp() if bars is None else bars
    o = bars.open.to_numpy()
    return pd.DataFrame([{"trade_id": 1, "direction": direction, "entry_idx": entry_idx, "exit_idx": exit_idx,
                          "entry_price": o[entry_idx] if entry_price is None else entry_price,
                          "exit_price": o[exit_idx] if exit_price is None else exit_price}])


def _report_and_bars(n=60, planted_edge=True, seed_bars=30, seed_trades=31):
    bars = make_bars(n=4000, seed=seed_bars)
    trades = make_trades(bars, n=n, seed=seed_trades, planted_edge=planted_edge)
    return parse_report(trades_to_report_text(trades)), load_bars(bars_to_ts_text(bars)), trades


def test_k0_equals_the_report():
    bars = make_bars(n=4000, seed=11)
    t = make_trades(bars, n=80, seed=12)
    open_ = bars.open.to_numpy()
    for leg in ("entry", "exit"):
        c = delay_curve(t, open_, PV, leg, max_k=5)
        p0 = c.points[0]
        assert np.array_equal(c.per_trade_pct[:, 0], pct_returns(t))
        assert p0.total_usd == usd_per_contract(t, PV).sum()
        assert p0.n_alive == len(t) and p0.n_skipped == 0 and p0.retention == 1.0
        pct, pts, alive = delayed_returns(t, open_, 0, leg)
        assert alive.all() and np.array_equal(pts * PV, usd_per_contract(t, PV))


@pytest.mark.parametrize("direction", [1, -1])
def test_linear_ramp_exact_values(direction):
    bars = _ramp()
    t = _trade(10, 20, direction=direction)
    assert t.entry_price[0] == 110.0 and t.exit_price[0] == 120.0
    o = bars.open.to_numpy()
    for k in range(1, 10):
        pct, pts, alive = delayed_returns(t, o, k, "entry")
        assert alive[0] and pct[0] == pytest.approx(direction * (120 / (110 + k) - 1))
        assert pts[0] * PV == pytest.approx(direction * (120 - (110 + k)) * PV)
        pct, pts, alive = delayed_returns(t, o, k, "exit")
        assert alive[0] and pct[0] == pytest.approx(direction * ((120 + k) / 110 - 1))
        assert pts[0] * PV == pytest.approx(direction * ((120 + k) - 110) * PV)
    c = delay_curve(t, o, PV, "exit", max_k=9)
    assert [p.total_usd for p in c.points] == pytest.approx([direction * (10 + k) * PV for k in range(10)])


def test_entry_delay_skips_trades_past_their_exit():
    t = _trade(10, 13)
    c = delay_curve(t, _ramp().open.to_numpy(), PV, "entry", max_k=10)
    assert [p.n_alive for p in c.points] == [1, 1, 1] + [0] * 8
    assert [p.n_skipped for p in c.points] == [0, 0, 0] + [1] * 8
    assert not np.isnan(c.per_trade_pct[0, :3]).any() and np.isnan(c.per_trade_pct[0, 3:]).all()
    for p in c.points[3:]:
        assert p.total_usd == 0.0
        assert np.isnan(p.mean_usd) and np.isnan(p.mean_pct) and np.isnan(p.win_rate) and np.isnan(p.profit_factor)
    assert c.first_nonpositive_k is None, "running out of trades must not read as the profit turning"


def test_exit_delay_skips_trades_past_the_series_end():
    bars = _ramp(n=30)
    t = _trade(20, 28, bars=bars)
    c = delay_curve(t, bars.open.to_numpy(), PV, "exit", max_k=5)
    assert [p.n_alive for p in c.points] == [1, 1, 0, 0, 0, 0]
    assert c.points[1].total_usd == pytest.approx((129 - 120) * PV)


def test_fixed_hold_mode():
    bars = _ramp(n=30)
    o = bars.open.to_numpy()
    t = pd.concat([_trade(10, 13, bars=bars), _trade(5, 25, direction=-1, bars=bars)], ignore_index=True)
    d, e, x = t.direction.to_numpy(), t.entry_idx.to_numpy(), t.exit_idx.to_numpy()
    for k in range(1, 8):
        pct, _, alive = delayed_returns(t, o, k, "entry", mode="fixed_hold")
        want_alive = x + k <= len(o) - 1
        assert np.array_equal(alive, want_alive)
        for i in np.flatnonzero(want_alive):
            assert pct[i] == pytest.approx(d[i] * (o[x[i] + k] / o[e[i] + k] - 1))
        assert np.isnan(pct[~want_alive]).all()
    a = delay_curve(t, o, PV, "exit", mode="fixed_exit", max_k=7)
    b = delay_curve(t, o, PV, "exit", mode="fixed_hold", max_k=7)
    assert _jsonable(a.points) == _jsonable(b.points)  # NaN-safe: NaN -> None on both sides
    assert np.array_equal(a.per_trade_pct, b.per_trade_pct, equal_nan=True)


def test_first_nonpositive_k():
    o = _ramp().open.to_numpy()
    assert delay_curve(_trade(10, 20), o, PV, "entry", max_k=10).first_nonpositive_k is None
    c = delay_curve(_trade(10, 20, exit_price=111.0), o, PV, "entry", max_k=10)
    assert c.points[1].total_usd == 0.0 and c.per_trade_pct[0, 1] == 0.0
    assert c.first_nonpositive_k == 1


def test_retention_is_nan_when_the_baseline_is_not_positive():
    o = _ramp().open.to_numpy()
    for leg in ("entry", "exit"):
        c = delay_curve(_trade(10, 20, direction=-1), o, PV, leg, max_k=5)
        assert c.points[0].total_usd < 0
        assert all(np.isnan(p.retention) for p in c.points)
    c = delay_curve(_trade(10, 20), o, PV, "entry", max_k=3)
    assert [p.retention for p in c.points] == pytest.approx([1.0, 9 / 10, 8 / 10, 7 / 10])


def test_validation_errors():
    o = _ramp().open.to_numpy()
    t = _trade(10, 20)
    for bad in (0, -1, 2.5, "3", True):
        with pytest.raises(ValueError, match="max_k"):
            delay_curve(t, o, PV, "entry", max_k=bad)
    with pytest.raises(ValueError, match="mode"):
        delay_curve(t, o, PV, "entry", mode="fixed")
    with pytest.raises(ValueError, match="leg"):
        delay_curve(t, o, PV, "both")
    with pytest.raises(ValueError, match="leg"):
        delayed_returns(t, o, 1, "both")
    with pytest.raises(ValueError, match="k must be"):
        delayed_returns(t, o, -1, "entry")
    rep, bars, _ = _report_and_bars(n=40)
    with pytest.raises(ValueError, match="max_k"):
        run_timing(rep, bars, max_k=0)
    with pytest.raises(ValueError, match="mode"):
        run_timing(rep, bars, mode="nope")


def test_run_timing_raises_validation_failed_when_trades_are_not_in_the_bars():
    bars = make_bars(n=3000, seed=34)
    trades = make_trades(bars, n=40, seed=35)
    rep = parse_report(trades_to_report_text(trades))
    short_bars = load_bars(bars_to_ts_text(bars.iloc[: len(bars) // 2]))
    with pytest.raises(ValidationFailed) as ei:
        run_timing(rep, short_bars)
    assert any(c.name == "timestamps_found" and not c.passed for c in ei.value.checks)


def test_run_timing_end_to_end():
    rep, bars, trades = _report_and_bars()
    r = run_timing(rep, bars, max_k=10)
    assert r.meta["n_trades"] == len(trades) and r.meta["point_value"] == PV and r.meta["symbol"] == "@MNQ"
    assert r.meta["max_k"] == 10 and r.meta["mode"] == "fixed_exit"
    assert r.at_open_share == 1.0  # make_trades fills at the bar open
    for c in (r.entry, r.exit):
        assert len(c.points) == 11 and c.per_trade_pct.shape == (len(trades), 11)
        assert c.points[0].n_alive == len(trades)
        assert c.points[0].total_usd == pytest.approx(
            (trades.direction * (trades.exit_price - trades.entry_price) * PV).sum())
    # the planted edge exits at the best open in hindsight, so a later exit must give some of it back
    assert r.exit.points[1].total_usd < r.exit.points[0].total_usd


def test_to_json():
    rep, bars, _ = _report_and_bars(n=40)
    r = run_timing(rep, bars, max_k=4)
    s = to_json(r)
    d = json.loads(s)
    json.dumps(d, allow_nan=False)
    assert set(d) == {f for f in TimingResult.__dataclass_fields__}
    assert len(d["entry"]["points"]) == 5 and d["entry"]["kind"] == "entry" and d["exit"]["kind"] == "exit"
    assert isinstance(d["meta"]["first_entry"], str)
    t = _trade(10, 20)
    c = delay_curve(t, _ramp().open.to_numpy(), PV, "entry", max_k=12)
    assert c.points[0].profit_factor == float("inf")
    dc = json.loads(json.dumps(_jsonable(c), allow_nan=False))
    assert dc["points"][0]["profit_factor"] is None
    assert dc["per_trade_pct"][0][11] is None  # skipped trade -> null


def test_deterministic_and_no_mutation():
    rep, bars, _ = _report_and_bars(n=40)
    before = rep.trades.copy()
    a, b = run_timing(rep, bars, max_k=5), run_timing(rep, bars, max_k=5)
    pd.testing.assert_frame_equal(rep.trades, before)
    assert to_json(a) == to_json(b)
    bars2 = make_bars(n=4000, seed=11)
    t = make_trades(bars2, n=30, seed=12)
    t_before = t.copy()
    delay_curve(t, bars2.open.to_numpy(), PV, "entry", mode="fixed_hold", max_k=5)
    pd.testing.assert_frame_equal(t, t_before)


@pytest.mark.parametrize("direction", [1, -1])
def test_earlier_shift_exact_values(direction):
    o = _ramp().open.to_numpy()
    t = _trade(10, 20, direction=direction)
    for k in range(1, 10):
        pct, pts, alive = delayed_returns(t, o, k, "entry", early=True)
        assert alive[0] and pct[0] == pytest.approx(direction * (120 / (110 - k) - 1))
        assert pts[0] == pytest.approx(direction * (120 - (110 - k)))
        pct, pts, alive = delayed_returns(t, o, k, "exit", early=True)
        assert alive[0] and pct[0] == pytest.approx(direction * ((120 - k) / 110 - 1))
    _, _, alive = delayed_returns(t, o, 10, "exit", early=True)
    assert not alive[0], "an exit moved back onto its entry bar is skipped"
    c = delay_curve(t, o, PV, "entry", max_k=10, early=True)
    assert c.shift == "earlier" and c.kind == "entry"
    assert [p.n_alive for p in c.points] == [1] * 11  # entry 10 - 10 = bar 0 still exists
    assert [p.total_usd for p in c.points] == pytest.approx([direction * (10 + k) * PV for k in range(11)])


def test_earlier_skip_rules():
    o = _ramp().open.to_numpy()
    c = delay_curve(_trade(3, 20), o, PV, "entry", max_k=6, early=True)
    assert [p.n_alive for p in c.points] == [1, 1, 1, 1, 0, 0, 0], "entry before the first bar is skipped"
    c = delay_curve(_trade(10, 13), o, PV, "exit", max_k=5, early=True)
    assert [p.n_alive for p in c.points] == [1, 1, 1, 0, 0, 0]
    assert c.first_nonpositive_k is None


def test_earlier_fixed_hold_and_k0():
    bars = make_bars(n=4000, seed=11)
    t = make_trades(bars, n=60, seed=12)
    o = bars.open.to_numpy()
    for leg in ("entry", "exit"):
        c = delay_curve(t, o, PV, leg, max_k=3, early=True)
        assert np.array_equal(c.per_trade_pct[:, 0], pct_returns(t)) and c.points[0].retention == 1.0
    e, x, d = t.entry_idx.to_numpy(), t.exit_idx.to_numpy(), t.direction.to_numpy()
    for k in (1, 5):
        pct, _, alive = delayed_returns(t, o, k, "entry", mode="fixed_hold", early=True)
        assert alive.all()  # make_trades never enters before bar 10
        assert pct == pytest.approx(d * (o[x - k] / o[e - k] - 1))
    a = delay_curve(t, o, PV, "exit", mode="fixed_exit", max_k=4, early=True)
    b = delay_curve(t, o, PV, "exit", mode="fixed_hold", max_k=4, early=True)
    assert _jsonable(a.points) == _jsonable(b.points)


def test_run_timing_has_earlier_curves():
    rep, bars, trades = _report_and_bars()
    r = run_timing(rep, bars, max_k=6)
    for c, kind in ((r.entry_early, "entry"), (r.exit_early, "exit")):
        assert c.shift == "earlier" and c.kind == kind and len(c.points) == 7
        assert c.points[0].total_usd == r.entry.points[0].total_usd
    assert r.entry.shift == "later" and r.exit.shift == "later"
    # planted-edge exits sit at the best open in hindsight: moving them earlier must give some back too
    assert r.exit_early.points[1].mean_usd < r.exit.points[0].mean_usd
