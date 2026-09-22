import numpy as np
import pytest

from robustness.multiwalk_text import parse_multiwalk_text
from robustness.selection import effective_trials, reality_check, selection_test, stationary_bootstrap_indices
from robustness.synthetic_multiwalk import make_multiwalk, make_walkforward_db
from robustness.walkforward_db import parse_walkforward_db
from robustness.windows import derive_windows


def test_bootstrap_indices_are_valid_blocks():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(500, rng, mean_block=20.0)
    assert idx.shape == (500,) and idx.min() >= 0 and idx.max() < 500
    steps = np.diff(idx)
    continuations = ((steps == 1) | (steps == -499)).mean()
    assert 0.9 < continuations < 0.99          # ~1 - 1/20 of steps continue the block


def test_effective_trials_bounds():
    rng = np.random.default_rng(1)
    indep = rng.normal(size=(10, 2000))
    assert 7.0 < effective_trials(indep) <= 10.0
    common = np.tile(rng.normal(size=(1, 2000)), (10, 1)) + 0.01 * rng.normal(size=(10, 2000))
    assert effective_trials(common) < 1.5
    assert effective_trials(np.zeros((3, 50))) == 0.0


def test_reality_check_on_noise_is_not_significant_and_on_planted_best_is():
    rng = np.random.default_rng(2)
    noise = rng.normal(0.0, 100.0, size=(40, 800))
    w = reality_check(noise, pick_index=int(np.argmax(noise.mean(axis=1))), n_boot=200, mean_block=20.0, seed=0)
    assert w.p_best > 0.2 and w.p_pick == w.p_best and w.best_index == w.pick_index
    planted = noise.copy(); planted[7] += 25.0
    w2 = reality_check(planted, pick_index=7, n_boot=200, mean_block=20.0, seed=0)
    assert w2.best_index == 7 and w2.p_best < 0.05 and w2.p_pick < 0.05
    assert 1.0 <= w2.n_eff <= 40.0 and w2.null_max.shape == (200,)


def test_selection_test_runs_per_complete_window_and_is_deterministic():
    text, sched = make_multiwalk({"A": list(range(5)), "B": list(range(4))}, n_days=900, seed=3, structure="persistent", split=600)
    grid = parse_multiwalk_text(text)
    windows = derive_windows(parse_walkforward_db(make_walkforward_db(sched))[0], grid.dates)
    r1 = selection_test(grid, windows, n_boot=100, seed=0); r2 = selection_test(grid, windows, n_boot=100, seed=0)
    assert len(r1.windows) == 1 and r1.windows[0].complete and r1.windows[0].n_iter == 20
    assert r1.windows[0].p_best == r2.windows[0].p_best and r1.n_boot == 100 and r1.block_len == 20.0
    assert r1.windows[0].pick_index == windows[0].grid_row - 1
