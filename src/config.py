"""Configuration for USA Private Markets macro model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


ASSET_TICKERS = [
    {"ticker": "PSP", "group": "PE"},
    {"ticker": "BX", "group": "PE"},
    {"ticker": "KKR", "group": "PE"},
    {"ticker": "APO", "group": "PE"},
    {"ticker": "CG", "group": "PE"},
    {"ticker": "VBR", "group": "PE"},
    {"ticker": "HYG", "group": "PC"},
    {"ticker": "JNK", "group": "PC"},
    {"ticker": "BKLN", "group": "PC"},
    {"ticker": "SRLN", "group": "PC"},
    {"ticker": "BIZD", "group": "PC"},
    {"ticker": "ARCC", "group": "PC"},
    {"ticker": "SPY", "group": "BENCHMARK"},
    {"ticker": "IEF", "group": "BENCHMARK"},
]

FRED_SERIES = {
    "growth": [
        "A191RL1Q225SBEA",
        "INDPRO",
        "PAYEMS",
        "RRSFS",
    ],
    "labour": [
        "UNRATE",
        "CIVPART",
    ],
    "inflation": [
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
    ],
    "rates": [
        "FEDFUNDS",
        "DGS2",
        "DGS10",
    ],
    "credit": [
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "STLFSI4",
    ],
    "risk": [
        "VIXCLS",
    ],
    "leading_indicators": [
        # Note: ISM PMI not directly in FRED - would need alternative source
        # Using alternative: Chicago Fed National Activity Index (CFNAI) as proxy
        "CFNAI",     # Chicago Fed National Activity Index (leading indicator)
        "UMCSENT",   # Consumer Confidence (University of Michigan)
    ],
    "cross_asset": [
        "DTWEXBGS",  # Trade-Weighted US Dollar Index
    ],
}


def get_all_tickers() -> List[str]:
    return [entry["ticker"] for entry in ASSET_TICKERS]


def get_ticker_metadata() -> List[Dict[str, str]]:
    return ASSET_TICKERS


def get_fred_series_ids() -> Dict[str, List[str]]:
    return FRED_SERIES

