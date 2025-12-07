#!/usr/bin/env python3
"""
Script 3: Run Correlation Analysis
Computes static and rolling correlations and generates visualizations.

Usage:
    python scripts/03_run_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis import compute_static_correlations, compute_rolling_correlations
from src.plotting import plot_corr_heatmap, plot_rolling_corr, plot_scatter_with_regression

def main():
    print("=" * 60)
    print("STEP 3: Running Correlation Analysis")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed data...")
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", index_col=0, parse_dates=True)
    
    # Add group column for plotting
    from src.config import ASSET_TICKERS
    groups = {entry["ticker"]: entry["group"] for entry in ASSET_TICKERS}
    returns["group"] = returns["ticker"].map(groups)
    
    # Compute static correlations
    print("\nComputing static correlations...")
    corr_lag0 = compute_static_correlations(returns, features, lag=0)
    corr_lag1 = compute_static_correlations(returns, features, lag=1)
    
    # Save correlation tables
    reports_dir = Path("reports/tables")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    corr_lag0.to_csv(reports_dir / "static_correlations_lag0.csv", index=False)
    corr_lag1.to_csv(reports_dir / "static_correlations_lag1.csv", index=False)
    print(f"  - Saved correlation tables to {reports_dir}")
    
    # Generate heatmaps
    print("\nGenerating correlation heatmaps...")
    corr_with_groups = corr_lag1.merge(returns[["ticker", "group"]].drop_duplicates(), 
                                       left_on="asset", right_on="ticker", how="left")
    
    for group in ["PE", "PC", "BENCHMARK"]:
        path = plot_corr_heatmap(corr_with_groups, group, lag=1)
        print(f"  - Created {path}")
    
    # Generate rolling correlations for key relationships
    print("\nComputing rolling correlations (36-month window)...")
    key_pairs = [
        ("PSP", "a191rl1q225sbea_level_z", "PSP vs GDP Growth"),
        ("BX", "bamlh0a0hym2_diff_z", "BX vs HY OAS"),
        ("HYG", "stlfsi4_diff_z", "HYG vs Financial Stress"),
        ("BIZD", "fedfunds_diff_z", "BIZD vs Fed Funds"),
    ]
    
    for asset, feature, title in key_pairs:
        rolling = compute_rolling_correlations(returns, features, asset, feature, window=36)
        if not rolling.empty:
            path = plot_rolling_corr(rolling, title)
            print(f"  - Created {path}")
    
    # Generate scatter plots
    print("\nGenerating scatter plots...")
    combined = returns.set_index("date").pivot_table(
        index="date", columns="ticker", values="return"
    ).join(features)
    
    for asset, feature, _ in key_pairs[:4]:  # Use same key pairs
        if asset in combined.columns and feature in combined.columns:
            asset_series = combined[asset].dropna()
            feature_series = combined[feature].dropna()
            aligned = pd.concat([asset_series, feature_series], axis=1).dropna()
            if len(aligned) > 30:  # Only plot if enough data
                path = plot_scatter_with_regression(
                    aligned[asset], aligned[feature], asset, feature
                )
                print(f"  - Created {path}")
    
    print("\nAnalysis complete!")
    print("\nGenerated outputs:")
    print(f"  - Tables: {reports_dir}")
    print(f"  - Figures: {Path('reports/figures')}")
    print("\nNext: Review correlation reports in reports/ directory")

if __name__ == "__main__":
    main()

