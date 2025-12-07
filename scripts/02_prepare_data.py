#!/usr/bin/env python3
"""
Script 2: Prepare Data
Transforms raw data into monthly returns and engineered features.

Usage:
    python scripts/02_prepare_data.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prep_data import (
    build_monthly_returns,
    build_monthly_macro,
    make_feature_matrix,
    save_processed_data
)

def main():
    print("=" * 60)
    print("STEP 2: Preparing Data")
    print("=" * 60)
    
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

if __name__ == "__main__":
    main()

