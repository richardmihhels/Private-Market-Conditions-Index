"""Correlation analysis helpers."""

from __future__ import annotations

from typing import List

import pandas as pd


def _prepare_panel(returns_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    ret = returns_df.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret = ret.pivot_table(index="date", columns="ticker", values="return")
    features_df = features_df.copy()
    features_df.index = pd.to_datetime(features_df.index)
    combined = ret.join(features_df, how="inner")
    return combined


def compute_static_correlations(
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    lag: int = 0,
) -> pd.DataFrame:
    combined = _prepare_panel(returns_df, features_df)
    results = []
    feature_cols = features_df.columns
    asset_cols = returns_df["ticker"].unique()
    for asset in asset_cols:
        asset_series = combined[asset]
        if lag:
            feature_data = combined[feature_cols].shift(lag)
        else:
            feature_data = combined[feature_cols]
        aligned = asset_series.to_frame().join(feature_data).dropna()
        if aligned.empty:
            continue
        for feature in feature_cols:
            pearson = aligned[asset].corr(aligned[feature])
            spearman = aligned[asset].corr(aligned[feature], method="spearman")
            results.append(
                {
                    "asset": asset,
                    "feature": feature,
                    "lag": lag,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(results)


def compute_rolling_correlations(
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    asset: str,
    feature: str,
    window: int = 36,
    lag: int = 0,
) -> pd.Series:
    combined = _prepare_panel(returns_df, features_df)
    asset_series = combined[asset]
    feature_series = combined[feature]
    if lag:
        feature_series = feature_series.shift(lag)
    aligned = pd.concat([asset_series, feature_series], axis=1).dropna()
    rolling = aligned[asset].rolling(window).corr(aligned[feature])
    rolling.name = f"{asset}_{feature}_rolling_corr"
    return rolling.dropna()



