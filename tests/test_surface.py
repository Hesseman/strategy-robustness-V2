import numpy as np
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.surface import daily_drawdown_stats, metric_values, window_metrics

from conftest import FIXTURES


def test_hand_traced_drawdown_stats():
    # equity 10, -20, -15, 25; peak 10,10,10,25; drawdown 0,-30,-25,0
    net, max_dd, avg_dd = daily_drawdown_stats(np.array([10.0, -30.0, 5.0, 40.0]))
    assert (net, max_dd, avg_dd) == (25.0, -30.0, -13.75)


def test_flat_start_drawdown_is_counted():
    net, max_dd, avg_dd = daily_drawdown_stats(np.array([-5.0, -5.0, 20.0]))
    assert max_dd == -10.0 and net == 10.0 and avg_dd == pytest.approx(-5.0)


def test_empty_is_zero():
    assert daily_drawdown_stats(np.array([])) == (0.0, 0.0, 0.0)


def test_window_metrics_on_the_mini_grid():
    g = parse_multiwalk_text((FIXTURES / "mini_multiwalk.txt").read_text(encoding="utf-8"))
    mask = np.array([True, True, True, False, False])
    m = window_metrics(g, mask)
    assert m.n_days == 3
    assert m.net_profit.tolist() == [30.0, -20.0, 0.0, 15.0]
    assert m.max_dd.tolist() == [-20.0, -20.0, 0.0, 0.0]
    assert m.avg_dd[0] == pytest.approx(-20.0 / 3) and m.avg_dd[2] == 0.0
    assert m.np_over_avgdd[0] == pytest.approx(30.0 / (20.0 / 3)) and np.isnan(m.np_over_avgdd[2])
    assert m.n_trades.tolist() == [1, 1, 0, 0]
    assert metric_values(m, "NP").tolist() == m.net_profit.tolist()
    assert np.array_equal(metric_values(m, "NPAvgDD"), m.np_over_avgdd, equal_nan=True)
    with pytest.raises(ValueError):
        metric_values(m, "Sharpe")
