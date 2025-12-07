#!/usr/bin/env python3
"""
Script 1: Download Data
Downloads price data from Yahoo Finance and macro data from FRED.

Usage:
    python scripts/01_download_data.py
    
Requirements:
    - FRED_API_KEY environment variable must be set
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.download_data import main

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Downloading Data")
    print("=" * 60)
    main(
        data_dir="data",
        price_start="2005-01-01",
        fred_start="2000-01-01"
    )
    print("\nData download complete!")
    print("Output files:")
    print("  - data/raw/prices/prices.csv")
    print("  - data/raw/fred/fred_series.csv")

