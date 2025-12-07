# Quick Start Guide

## 1. First Time Setup (5 minutes)

```bash
# Get your FREE FRED API key
# Visit: https://fred.stlouisfed.org/docs/api/api_key.html

# Clone and setup
git clone https://github.com/richardmihhels/Private-Market-Conditions-Index.git
cd Private-Market-Conditions-Index

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your FRED API key
export FRED_API_KEY='your_key_here'  # Windows: set FRED_API_KEY=your_key_here
```

## 2. Run the Complete Analysis (10-15 minutes)

```bash
python scripts/run_full_pipeline.py
```

That's it! The pipeline will:
1. Download 15+ years of price and macro data
2. Process into monthly returns and features
3. Run correlation analysis
4. Generate all visualizations

## 3. View Results

### Quick Overview
- **Executive Summary:** `reports/EXECUTIVE_SUMMARY.md`
- **Detailed Analysis:** `reports/CORRELATION_ANALYSIS_REPORT.md`

### Visualizations
- **Heatmaps:** `reports/figures/corr_heatmap_*.png`
- **Rolling Correlations:** `reports/figures/rolling_corr_*.png`
- **Scatter Plots:** `reports/figures/scatter_*.png`

### Data Tables
- **Static Correlations:** `reports/tables/static_correlations_lag*.csv`
- **Key Findings:** `reports/tables/key_findings_summary.csv`

### Interactive Analysis
```bash
jupyter notebook notebooks/usa_pm_macro_correlations.ipynb
```

## 4. Understanding the Results

### What to Look For

**Strong Positive Correlations (+0.4 to +1.0)**
- These macro factors move *with* asset returns
- Example: Private equity up when GDP growth is strong

**Strong Negative Correlations (-1.0 to -0.4)**
- These macro factors move *against* asset returns
- Example: Private credit down when credit spreads widen

**Near Zero Correlations (-0.2 to +0.2)**
- Weak or no relationship
- May still be interesting if they're stable or regime-dependent

### Key Assets Groups

- **PE (Private Equity):** PSP, BX, KKR, APO, CG, VBR
- **PC (Private Credit):** HYG, JNK, BKLN, SRLN, BIZD, ARCC
- **Benchmarks:** SPY, IEF

### Key Macro Features

- **Growth:** GDP, payrolls, industrial production
- **Credit:** HY spreads, IG spreads, financial stress
- **Rates:** Fed funds, 2Y/10Y yields, yield curve
- **Sentiment:** VIX, consumer confidence

## 5. Common Tasks

### Update Data (Monthly Refresh)
```bash
python scripts/01_download_data.py
python scripts/02_prepare_data.py
python scripts/03_run_analysis.py
```

### Add a New Asset
Edit `src/config.py`:
```python
ASSET_TICKERS = [
    {"ticker": "NEW_TICKER", "group": "PE"},  # or "PC" or "BENCHMARK"
    # ... existing tickers
]
```
Then re-run the pipeline.

### Add a New Macro Variable
1. Find FRED series ID: https://fred.stlouisfed.org/
2. Edit `src/config.py`:
```python
FRED_SERIES = {
    "your_category": ["FRED_SERIES_ID"],
    # ... existing series
}
```
3. Edit `src/prep_data.py` to specify transformation (YoY, diff, or level)
4. Re-run the pipeline

### Change Analysis Parameters
```python
# In scripts/03_run_analysis.py

# Rolling correlation window (default: 36 months)
rolling = compute_rolling_correlations(returns, features, asset, feature, window=24)

# Lag structure (default: 0 and 1)
corr_lag2 = compute_static_correlations(returns, features, lag=2)
```

## 6. Troubleshooting

### "FRED_API_KEY not found"
```bash
# Make sure you've exported the key in your current terminal session
export FRED_API_KEY='your_key_here'

# Or add to your ~/.bashrc or ~/.zshrc for persistence
echo "export FRED_API_KEY='your_key_here'" >> ~/.zshrc
```

### "No data for ticker X"
- Some ETFs have limited history
- Check Yahoo Finance manually to verify ticker symbol
- Consider replacing with alternative ticker

### "Missing values in analysis"
- This is normal - series have different start dates
- The pipeline automatically aligns on common dates
- Check `data_processed/features_monthly.csv` for date range

### Permission errors with .venv directories
- These are harmless and can be ignored
- .venv directories are not tracked in git
- They contain Python packages from virtual environment

## 7. Project Structure Overview

```
Key Files:
├── scripts/
│   ├── run_full_pipeline.py      ← Run this first!
│   ├── 01_download_data.py
│   ├── 02_prepare_data.py
│   └── 03_run_analysis.py
│
├── src/                           ← Core modules (don't run directly)
│   ├── config.py                  ← Edit to add assets/variables
│   ├── download_data.py
│   ├── prep_data.py              ← Edit to change transformations
│   ├── analysis.py
│   └── plotting.py
│
├── reports/                       ← View results here
│   ├── EXECUTIVE_SUMMARY.md      ← Start here
│   ├── figures/                   ← Charts and plots
│   └── tables/                    ← CSV files with numbers
│
├── notebooks/                     ← Interactive analysis
│   └── usa_pm_macro_correlations.ipynb
│
├── DATA_DICTIONARY.md            ← Explains all variables
├── README.md                     ← Full documentation
└── requirements.txt              ← Python dependencies
```

## 8. Next Steps

After reviewing the initial results:

1. **Identify top drivers** - Which macro variables show strongest correlations?
2. **Check stability** - Look at rolling correlation plots for time variation
3. **Economic intuition** - Do the relationships make sense?
4. **Regime analysis** - Manually inspect crisis vs. expansion periods
5. **Extend analysis** - Add assets, variables, or time periods of interest

## 9. Learning Resources

- **Data Dictionary:** `DATA_DICTIONARY.md` - understand every variable
- **Full Documentation:** `README.md` - complete methodology
- **CRISP-DM Report:** `H5_report.pdf` - project background and goals
- **Code Comments:** All modules have detailed docstrings

## 10. Getting Help

Check the documentation in order:
1. This QUICKSTART.md (you are here)
2. DATA_DICTIONARY.md - variable definitions
3. README.md - full documentation
4. H5_report.pdf - project context

For code-specific questions, check the docstrings:
```python
from src import analysis
help(analysis.compute_static_correlations)
```

---

**Ready?** Run this now:
```bash
python scripts/run_full_pipeline.py
```

Then open `reports/EXECUTIVE_SUMMARY.md` to see your results!




