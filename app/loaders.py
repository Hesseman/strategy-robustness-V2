"""Cached decoding of the two uploads, shared by the five cards and the timing section, so an
uploaded file is decoded and parsed once and both sections read it under the same rule.

Only reason to change: how an uploaded TradeStation export is decoded before parsing."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from robustness.bars_loader import load_bars
from robustness.report_parser import ParsedReport, parse_report


@st.cache_data(show_spinner=False)
def parsed_report(report_bytes: bytes) -> ParsedReport:
    """Accepts the raw bytes of a Strategy Performance Report export; returns parse_report of
    the text decoded as UTF-8 with an optional BOM (undecodable bytes replaced). Guarantees
    one decode rule for every section; raises ReportFormatError like parse_report; cached per
    distinct bytes (Streamlit hands each caller its own copy)."""
    return parse_report(report_bytes.decode("utf-8-sig", errors="replace"))


@st.cache_data(show_spinner=False)
def loaded_bars(bars_bytes: bytes) -> pd.DataFrame:
    """Accepts the raw bytes of a Data Window bar export; returns load_bars of the text decoded
    as UTF-8 with an optional BOM (undecodable bytes replaced). Guarantees one decode rule for
    every section; raises BarsFormatError like load_bars; cached per distinct bytes (Streamlit
    hands each caller its own copy)."""
    return load_bars(bars_bytes.decode("utf-8-sig", errors="replace"))
