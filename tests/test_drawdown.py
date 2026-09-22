import numpy as np
import pandas as pd
import pytest

from robustness.drawdown import cdar, drawdown_analysis, drawdown_episodes, margin_check, sharpe, sortino

PNL = np.array([100.0, -50.0, -70.0, 30.0, 200.0, -20.0, -300.0, 50.0, 100.0, 500.0])
# cum: 100, 50, -20, 10, 210, 190, -110, -60, 40, 540


def test_episodes_by_hand():
    eps = drawdown_episodes(PNL)
    assert len(eps) == 2
    assert eps[0] == {"peak_i": 0, "trough_i": 2, "recovery_i": 4, "depth": -120.0}
    assert eps[1] == {"peak_i": 4, "trough_i": 6, "recovery_i": 9, "depth": -320.0}


def test_unrecovered_episode():
    eps = drawdown_episodes(np.array([10.0, -5.0, -5.0]))
    assert len(eps) == 1 and eps[0]["recovery_i"] is None and eps[0]["depth"] == -10.0


def test_drawdown_from_flat_start_is_counted():
    eps = drawdown_episodes(np.array([-100.0, -100.0, 300.0, 50.0]))
    assert eps == [{"peak_i": -1, "trough_i": 1, "recovery_i": 2, "depth": -200.0}]
    times = pd.date_range("2020-01-01", periods=4, freq="7D")
    r = drawdown_analysis(np.array([-100.0, -100.0, 300.0, 50.0]), times - pd.Timedelta(days=1), times)
    assert r.max_dd == 200.0 and r.cdar80 == 200.0 and r.capital == 1000.0
    assert r.worst["peak_time"] == (times - pd.Timedelta(days=1)).min() and r.worst["trough_time"] == times[1]


def test_cdar():
    depths = np.array([120.0, 320.0])
    assert cdar(depths, 0.80) == 320.0                     # k = ceil(0.2*2) = 1 -> worst one
    assert cdar(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]), 0.80) == 95.0


def test_sharpe_sortino():
    years = 2.0
    assert np.isclose(sharpe(PNL, years), PNL.mean() / PNL.std(ddof=1) * np.sqrt(len(PNL) / years))
    dd = np.sqrt(np.mean(np.minimum(PNL, 0) ** 2))
    assert np.isclose(sortino(PNL, years), PNL.mean() / dd * np.sqrt(len(PNL) / years))


def test_drawdown_analysis_capital_and_metrics():
    times = pd.date_range("2020-01-01", periods=10, freq="73D")      # ~1.8 years span
    entry = times - pd.Timedelta(days=1)
    r = drawdown_analysis(PNL, pd.DatetimeIndex(entry), pd.DatetimeIndex(times))
    assert r.n_trades == 10 and r.n_episodes == 2 and r.n_unrecovered == 0
    assert r.max_dd == 320.0 and r.cdar80 == 320.0 and r.capital == 1600.0
    assert np.isclose(r.years, (times[-1] - entry[0]).days / 365.25)
    assert np.isclose(r.annual_usd, PNL.sum() / r.years)
    assert np.isclose(r.annual_pct, r.annual_usd / 1600.0)
    assert np.isclose(r.max_dd_pct_of_capital, 0.2)
    assert np.isclose(r.calmar, r.annual_usd / 320.0)
    assert np.isclose(r.profit_over_avg_dd, r.annual_usd / 220.0)
    assert r.worst["peak_time"] == times[4] and r.worst["trough_time"] == times[6] and r.worst["recovery_time"] == times[9]
    assert r.equity.tolist() == np.cumsum(PNL).tolist()


def test_drawdown_analysis_orders_by_exit_time():
    times = pd.DatetimeIndex(["2020-03-01", "2020-01-01", "2020-02-01"])
    r = drawdown_analysis(np.array([3.0, 1.0, 2.0]), times - pd.Timedelta(days=1), times)
    assert r.equity.tolist() == [1.0, 3.0, 6.0]


def test_margin_check():
    m = margin_check(1600.0, 200.0)
    assert m == {"today_margin_usd": 200.0, "capital_usd": 1600.0, "margin_to_equity": 0.125, "contracts_covered": 8.0}


def test_capital_override_changes_the_denominator_only():
    import numpy as np, pandas as pd
    from robustness.drawdown import drawdown_analysis
    pnl = np.array([100.0, -60.0, 80.0, -40.0, 120.0, -30.0])
    t = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=6, freq="7D"))
    a = drawdown_analysis(pnl, t, t + pd.Timedelta(days=1))
    b = drawdown_analysis(pnl, t, t + pd.Timedelta(days=1), capital_override=10_000.0)
    assert a.capital_basis == "5 x CDaR-80" and a.capital == a.capital_5x_cdar80 == 5 * a.cdar80
    assert b.capital_basis == "fixed" and b.capital == 10_000.0 and b.capital_5x_cdar80 == a.capital_5x_cdar80
    assert b.cdar80 == a.cdar80 and b.annual_usd == a.annual_usd
    assert b.annual_pct == pytest.approx(b.annual_usd / 10_000.0) and b.max_dd_pct_of_capital == pytest.approx(b.max_dd / 10_000.0)


def test_capital_override_must_be_positive():
    import numpy as np, pandas as pd
    from robustness.drawdown import drawdown_analysis
    t = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=3, freq="7D"))
    with pytest.raises(ValueError):
        drawdown_analysis(np.array([1.0, 2.0, 3.0]), t, t, capital_override=0.0)
