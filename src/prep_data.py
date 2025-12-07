"""Transform raw price and macro data into monthly datasets and features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .config import ASSET_TICKERS, get_fred_series_ids


def build_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    groups = {entry["ticker"]: entry["group"] for entry in ASSET_TICKERS}
    monthly = (
        df.set_index("date")
        .groupby("ticker")["adj_close"]
        .resample("M")
        .last()
        .groupby("ticker")
        .pct_change()
        .dropna()
        .reset_index()
        .rename(columns={"adj_close": "return"})
    )
    monthly["group"] = monthly["ticker"].map(groups)
    return monthly


def build_monthly_macro(fred_df: pd.DataFrame) -> pd.DataFrame:
    macro = fred_df.copy()
    macro.index = pd.to_datetime(macro.index)
    monthly = macro.resample("M").last()
    quarterly_ids = get_fred_series_ids()["growth"][:1]  # GDP series
    for sid in quarterly_ids:
        if sid in macro.columns:
            monthly[sid] = macro[sid].resample("Q").last().reindex(monthly.index, method="ffill")
    
    # Calculate yield curve slope (10Y - 2Y) if both exist
    if "DGS10" in monthly.columns and "DGS2" in monthly.columns:
        monthly["YIELD_CURVE_SLOPE"] = monthly["DGS10"] - monthly["DGS2"]
    
    return monthly


NEGATIVE_FEATURES = {"UNRATE", "BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "VIXCLS", "YIELD_CURVE_SLOPE"}


def make_feature_matrix(monthly_macro: pd.DataFrame) -> pd.DataFrame:
    features = {}
    for col in monthly_macro.columns:
        series = monthly_macro[col].copy()
        # YoY transformations for inflation and activity indicators
        if col in {"CPIAUCSL", "CPILFESL", "PCEPI", "INDPRO", "RRSFS"}:
            transformed = series.pct_change(12)
            name = f"{col.lower()}_yoy"
        # Diff transformations for rates, spreads, and indicators
        elif col in {"FEDFUNDS", "DGS2", "DGS10", "BAMLH0A0HYM2", "BAMLC0A0CM", "STLFSI4", "VIXCLS", "UNRATE", "YIELD_CURVE_SLOPE"}:
            transformed = series.diff()
            name = f"{col.lower()}_diff"
        # Level transformations for PMI, sentiment, and other level indicators
        elif col in {"NAPM", "UMCSENT", "DTWEXBGS"}:
            transformed = series
            name = f"{col.lower()}_level"
        else:
            transformed = series
            name = f"{col.lower()}_level"
        transformed = transformed.dropna()
        if col in NEGATIVE_FEATURES:
            transformed = -transformed
        z = (transformed - transformed.mean()) / transformed.std(ddof=0)
        features[name + "_z"] = z
    feature_df = pd.concat(features, axis=1).dropna(how="all")
    return feature_df


def save_processed_data(
    returns_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_dir: str = "data_processed",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    returns_df.to_csv(out / "monthly_returns.csv", index=False)
    macro_df.to_csv(out / "macro_monthly.csv", index=True)
    features_df.to_csv(out / "features_monthly.csv", index=True)

