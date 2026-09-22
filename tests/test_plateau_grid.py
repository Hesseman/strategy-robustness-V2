import numpy as np
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.plateau_grid import neighbours, plateau_score, plateau_test
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db
from robustness.windows import derive_windows


def test_neighbours_are_chebyshev_one():
    pos = np.array([[a, b] for b in range(3) for a in range(3)])   # 3x3, first axis fastest
    centre = 4                                                     # (1,1)
    assert sorted(neighbours(pos, centre).tolist()) == [0, 1, 2, 3, 5, 6, 7, 8]
    assert sorted(neighbours(pos, 0).tolist()) == [1, 3, 4]        # corner


def test_score_hand_traced():
    score, flat, share = plateau_score(np.array([10.0, 10.0, 10.0, 10.0]))
    assert (score, flat, share) == (1.0, 1.0, 1.0)
    score, flat, share = plateau_score(np.array([10.0, -5.0, -5.0, -5.0]))
    assert share == 0.0 and flat == 0.0 and score == 0.0          # spike: neighbourhood mean < 0
    score, flat, share = plateau_score(np.array([10.0, 8.0, 12.0, -1.0]))
    assert share == pytest.approx(2 / 3) and 0 < flat < 1 and score == pytest.approx(0.5 * flat + 0.5 * share)
    assert plateau_score(np.array([np.nan, 1.0])) == (0.0, 0.0, 0.0)


def test_persistent_surface_scores_higher_than_noise():
    from robustness.surface import metric_values, window_metrics
    out, pairs_by = {}, {}
    for s in ("persistent", "noise"):
        text, sched = make_multiwalk({"A": list(range(6)), "B": list(range(6))}, n_days=1200, seed=11, structure=s, split=800)
        grid = parse_multiwalk_text(text)
        windows = derive_windows(parse_walkforward_db(make_walkforward_db(sched))[0], grid.dates)
        pairs = [(metric_values(window_metrics(grid, w.is_mask), "NP"), metric_values(window_metrics(grid, w.oos_mask), "NP")) for w in windows]
        out[s] = plateau_test(pairs, windows, grid)
        pairs_by[s] = pairs
    assert out["persistent"].pooled_score > out["noise"].pooled_score
    w = out["persistent"].windows[0]
    assert w.n_neighbours >= 3 and 0.0 <= w.score_oos <= 1.0
    assert w.oos_values[0] == pytest.approx(pairs_by["persistent"][0][1][w.pick_index])   # pick comes first
    assert out["persistent"].n_complete == 1
