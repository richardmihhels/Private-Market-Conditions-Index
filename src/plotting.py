"""Plotting helpers for USA private markets correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_corr_heatmap(corr_df: pd.DataFrame, group: str, lag: int) -> Path:
    subset = corr_df[corr_df["lag"] == lag]
    if "group" in corr_df.columns:
        subset = subset[subset["group"] == group]
    pivot = subset.pivot(index="feature", columns="asset", values="pearson")
    plt.figure(figsize=(len(pivot.columns) * 0.8 + 4, len(pivot.index) * 0.3 + 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"Pearson correlations ({group}, lag={lag})")
    plt.tight_layout()
    path = FIG_DIR / f"corr_heatmap_{group}_lag{lag}.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_rolling_corr(series: pd.Series, title: str) -> Path:
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(title)
    plt.ylabel("Correlation")
    plt.tight_layout()
    fname = "_".join(title.lower().split()).replace("/", "_")
    path = FIG_DIR / f"rolling_corr_{fname}.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_scatter_with_regression(
    returns: pd.Series,
    feature: pd.Series,
    asset_name: str,
    feature_name: str,
) -> Path:
    aligned = pd.concat([returns, feature], axis=1).dropna()
    plt.figure(figsize=(6, 4))
    sns.regplot(x=feature_name, y=asset_name, data=aligned, scatter_kws={"alpha": 0.5})
    plt.title(f"{asset_name} vs {feature_name}")
    plt.tight_layout()
    fname = f"{asset_name}_{feature_name}".lower().replace(" ", "_")
    path = FIG_DIR / f"scatter_{fname}.png"
    plt.savefig(path)
    plt.close()
    return path



