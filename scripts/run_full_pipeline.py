#!/usr/bin/env python3
"""
Master Pipeline Script
Runs the complete analysis pipeline from data download to final reports.

Usage:
    python scripts/run_full_pipeline.py
    
Requirements:
    - FRED_API_KEY environment variable must be set
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.download_data import main as download_main
from src.prep_data import (
    build_monthly_returns,
    build_monthly_macro,
    make_feature_matrix,
    save_processed_data
)
from src.analysis import compute_static_correlations, compute_rolling_correlations
from src.plotting import plot_corr_heatmap, plot_rolling_corr, plot_scatter_with_regression
from src.config import ASSET_TICKERS


def run_step_1():
    """Step 1: Download Data"""
    print("\n" + "=" * 70)
    print("PIPELINE STEP 1: Download Data")
    print("=" * 70)
    
    download_main(
        data_dir="data",
        price_start="2005-01-01",
        fred_start="2000-01-01"
    )
    print("\nData download complete!")
    print("Output files:")
    print("  - data/raw/prices/prices.csv")
    print("  - data/raw/fred/fred_series.csv")


def run_step_2():
    """Step 2: Prepare Data"""
    print("\n" + "=" * 70)
    print("PIPELINE STEP 2: Prepare Data")
    print("=" * 70)
    
    # Load raw data
    print("\nLoading raw data...")
    prices = pd.read_csv("data/raw/prices/prices.csv")
    fred = pd.read_csv("data/raw/fred/fred_series.csv", index_col=0, parse_dates=True)
    
    # Build monthly returns
    print("Building monthly returns...")
    returns = build_monthly_returns(prices)
    print(f"  - Created {len(returns)} return observations across {returns['ticker'].nunique()} assets")
    
    # Build monthly macro
    print("Building monthly macro series...")
    macro_monthly = build_monthly_macro(fred)
    print(f"  - Created {len(macro_monthly)} monthly observations for {len(macro_monthly.columns)} series")
    
    # Engineer features
    print("Engineering features (YoY, diff, z-scores)...")
    features = make_feature_matrix(macro_monthly)
    print(f"  - Created {len(features.columns)} engineered features")
    
    # Save processed data
    print("\nSaving processed data...")
    save_processed_data(returns, macro_monthly, features, output_dir="data_processed")
    
    print("\nData preparation complete!")
    print("Output files:")
    print("  - data_processed/monthly_returns.csv")
    print("  - data_processed/macro_monthly.csv")
    print("  - data_processed/features_monthly.csv")


def run_step_3():
    """Step 3: Run Analysis"""
    print("\n" + "=" * 70)
    print("PIPELINE STEP 3: Run Correlation Analysis")
    print("=" * 70)
    
    # Load processed data
    print("\nLoading processed data...")
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", index_col=0, parse_dates=True)
    
    # Add group column for plotting
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
    
    for asset, feature, _ in key_pairs[:4]:
        if asset in combined.columns and feature in combined.columns:
            asset_series = combined[asset].dropna()
            feature_series = combined[feature].dropna()
            aligned = pd.concat([asset_series, feature_series], axis=1).dropna()
            if len(aligned) > 30:
                path = plot_scatter_with_regression(
                    aligned[asset], aligned[feature], asset, feature
                )
                print(f"  - Created {path}")
    
    print("\nAnalysis complete!")
    print(f"\nGenerated outputs:")
    print(f"  - Tables: {reports_dir}")
    print(f"  - Figures: {Path('reports/figures')}")


def main():
    print("=" * 70)
    print("PRIVATE MARKET CONDITIONS INDEX - FULL PIPELINE")
    print("=" * 70)
    print("\nThis pipeline will:")
    print("  1. Download price and macro data")
    print("  2. Process data into monthly returns and features")
    print("  3. Run correlation analysis and generate visualizations")
    print("\n" + "=" * 70)
    
    steps = [
        (1, "Download Data", run_step_1),
        (2, "Prepare Data", run_step_2),
        (3, "Run Analysis", run_step_3),
    ]
    
    for step_num, description, step_func in steps:
        try:
            step_func()
            print(f"\n✓ Step {step_num} completed successfully")
        except Exception as e:
            print(f"\n✗ Step {step_num} failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            print(f"\n✗ Pipeline stopped at step {step_num}")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated outputs:")
    print("  - data/raw/prices/prices.csv")
    print("  - data/raw/fred/fred_series.csv")
    print("  - data_processed/monthly_returns.csv")
    print("  - data_processed/macro_monthly.csv")
    print("  - data_processed/features_monthly.csv")
    print("  - reports/tables/*.csv")
    print("  - reports/figures/*.png")
    print("\nNext steps:")
    print("  - Review notebooks/usa_pm_macro_correlations.ipynb for detailed analysis")
    print("  - Check reports/CORRELATION_ANALYSIS_REPORT.md for findings")
    print("  - Review reports/figures/ for visualizations")


if __name__ == "__main__":
    main()

