import numpy as np
import pandas as pd
import pytest

from robustness.multiwalk_text import MultiWalkFormatError, parse_multiwalk_text

from conftest import FIXTURES


@pytest.fixture
def mini_text() -> str:
    return (FIXTURES / "mini_multiwalk.txt").read_text(encoding="utf-8")


def test_header_params_axes_and_grid_order(mini_text):
    g = parse_multiwalk_text(mini_text)
    assert g.param_names == ["A", "B"]
    assert g.params.tolist() == [[1, 10], [2, 10], [1, 20], [2, 20]]
    assert [a.tolist() for a in g.axes] == [[1, 2], [10, 20]]
    assert g.grid_pos.tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]
    assert g.shape == (2, 2) and g.n_iter == 4 and g.is_full_grid


def test_daily_series_and_dates(mini_text):
    g = parse_multiwalk_text(mini_text)
    assert list(g.dates) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]))
    assert g.daily_pnl.shape == (4, 5) and g.closed_pnl.shape == (4, 5)
    assert g.daily_pnl[0].tolist() == [0.0, 50.0, -20.0, 10.0, 0.0]
    assert g.closed_pnl[0].tolist() == [0.0, 0.0, 30.0, 0.0, 0.0]
    assert g.daily_pnl[3].sum() == 25.0


def test_closed_trades_per_iteration(mini_text):
    g = parse_multiwalk_text(mini_text)
    assert g.exit_dates[0].tolist() == [np.datetime64("2024-01-04")] and g.exit_pnl[0].tolist() == [30.0]
    assert g.exit_pnl[1].tolist() == [-20.0]
    assert len(g.exit_dates[2]) == 0 and len(g.exit_pnl[2]) == 0
    assert g.exit_dates[3].tolist() == [np.datetime64("2024-01-08")]


def test_crlf_and_bom_tolerated(mini_text):
    g = parse_multiwalk_text("﻿" + mini_text.replace("\n", "\r\n"))
    assert g.n_iter == 4


def test_missing_header_is_readable_error():
    with pytest.raises(MultiWalkFormatError, match="header"):
        parse_multiwalk_text("1,10~v1,1|20240102,0.00,0,0,0,0,0,0,0.00\n")


def test_wrong_param_count_names_the_iteration(mini_text):
    bad = mini_text.replace("\n2,10~", "\n2,10,3~", 1)
    with pytest.raises(MultiWalkFormatError, match="iteration 2"):
        parse_multiwalk_text(bad)


def test_differing_dates_names_the_iteration(mini_text):
    bad = mini_text.replace("|20240108,0.00,0,0,0,0,0,0,0.00~v2\n", "|20240109,0.00,0,0,0,0,0,0,0.00~v2\n", 1)
    with pytest.raises(MultiWalkFormatError, match="iteration 3"):
        parse_multiwalk_text(bad)


def test_missing_v1_tag_is_error(mini_text):
    bad = mini_text.replace("1,10~v1,1|", "1,10~|", 1)
    with pytest.raises(MultiWalkFormatError, match="v1"):
        parse_multiwalk_text(bad)
