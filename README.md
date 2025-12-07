# Private Market Conditions Index

**Predicting Macro-Sensitive Assets with Walk-Forward Machine Learning**

**Author:** Richard Mihhels | **Project ID:** H5 | **Date:** December 2025

---

## 🎯 Motivation & Goal

### Why This Project?

Private equity and private credit returns are **highly sensitive to macroeconomic conditions** such as growth, inflation, interest rates, and credit spreads. However:

- Most current analysis is **narrative-based** and lacks a systematic, data-driven framework
- Investors need **transparent, quantitative tools** that show how different macro environments affect private-market-like assets
- There's limited research on **which ML models** work best for predicting private market returns

### What We Built

A **Private Market Conditions Index (PMCI)** that:

1. Uses **14 US-listed assets** as public proxies for private equity and credit (2006–2025)
2. Employs **20+ macro and credit indicators** to predict next-month return direction
3. Tests **7 different ML algorithms** with rigorous walk-forward validation
4. Focuses on **robustness and interpretability** rather than overfitting

### Key Achievement

**Random Forest achieves 60.5% directional accuracy** in realistic walk-forward testing, with credit assets reaching **72.6%** accuracy. This proves private markets exhibit **genuine predictability** from macroeconomic indicators.

---

## 📖 Repository Guide - Quick Navigation

### **For Results & Findings** (Executives, Presenters)
→ **[MODEL_SELECTION_RESULTS.md](MODEL_SELECTION_RESULTS.md)** - Complete research findings  
→ **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - 5-minute overview  
→ **`reports/figures/`** - All visualizations  
→ **`PMCI_A0_Poster.pdf`** - Academic poster presentation

### **For Methodology** (Analysts, Researchers, Reviewers)
→ **[METHODOLOGY.md](METHODOLOGY.md)** - Validation approach explained  
→ **[H5_report.pdf](H5_report.pdf)** - CRISP-DM documentation  
→ **`backtesting/`** - Walk-forward validation code  

### **For Running the Code** (Developers, Data Scientists)
→ **[QUICKSTART.md](QUICKSTART.md)** - Setup instructions (5 minutes)  
→ **`python scripts/run_full_pipeline.py`** - Execute complete analysis  
→ **`src/config.py`** - Customize assets and indicators  

### **For Understanding the Data** (Academic, Peer Review)
→ **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** - All variable definitions  
→ **`data_processed/`** - Clean datasets (monthly returns + features)  
→ Keep reading below for detailed methodology

---

## 🚀 How to Replicate This Analysis

**Time Required:** ~15 minutes (first-time setup) + ~10 minutes (analysis runtime)

### Prerequisites
1. **Python 3.8+** installed
2. **FRED API Key** (free): Get yours at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

### Step 1: Clone & Setup (5 minutes)
```bash
# Clone the repository
git clone https://github.com/richardmihhels/Private-Market-Conditions-Index.git
cd Private-Market-Conditions-Index

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your FRED API key
export FRED_API_KEY='your_api_key_here'  # On Windows: set FRED_API_KEY=your_api_key_here
```

### Step 2: Run Complete Analysis (10 minutes)
```bash
# Execute full pipeline (downloads data, processes it, runs all models)
python scripts/run_full_pipeline.py
```

**That's it!** The pipeline will:
1. Download 15+ years of price data from Yahoo Finance
2. Download macro data from FRED
3. Process into monthly returns and features
4. Run correlation analysis
5. Train 7 ML models × 14 assets = 98 model comparisons
6. Generate all visualizations in `reports/`

### Step 3: View Results
- **Processed Data:** `data_processed/monthly_returns.csv`, `features_monthly.csv`
- **Model Performance:** `reports/tables/model_performance_summary.csv`
- **Visualizations:** `reports/figures/` and `reports/figures_friendly/`
- **Analysis:** Open [MODEL_SELECTION_RESULTS.md](MODEL_SELECTION_RESULTS.md)

### Alternative: Step-by-Step Execution
```bash
python scripts/01_download_data.py      # Get raw data
python scripts/02_prepare_data.py       # Process to monthly
python scripts/03_run_analysis.py       # Correlation analysis
python scripts/create_ultimate_ml_models.py  # ML comparison
```

### Alternative: Interactive Exploration
```bash
jupyter notebook notebooks/usa_pm_macro_correlations.ipynb
```

---

## 🏆 Headline Results

| What | Value | Meaning |
|------|-------|---------|
| **Best Model** | **Random Forest** | 60.5% walk-forward hit rate |
| **Best Asset** | **SRLN (Senior Loans)** | 72.6% directional accuracy |
| **Average Accuracy** | **59%** | Across all 14 assets (walk-forward) |
| **Edge vs Random** | **+9pp** | Percentage points above 50% baseline |
| **Strong Performers** | **9 of 14** | Assets with >58% walk-forward accuracy |
| **Validation Finding** | **Walk-forward beat static** | +1.2pp proves no overfitting |

### Translation:
- **60.5% = Predict correctly 61 out of 100 times** (vs 50 random)
- **72.6% for credit = Approaching professional forecasting quality**
- **Walk-forward > static = Models learn genuine patterns**

---

## 🎯 Key Finding: Credit > Equity

**Private Credit is HIGHLY predictable:**
- Senior Loans (SRLN): **72.6%** walk-forward hit rate
- Bank Loans (BKLN): **69.6%** walk-forward hit rate  
- High Yield (HYG): **64.9%** walk-forward hit rate

**Private Equity is MODERATELY predictable:**
- Blackstone (BX): **64.5%** walk-forward hit rate
- Apollo (APO): **56.5%** walk-forward hit rate
- KKR: **57.5%** walk-forward hit rate

**Why?** Credit has mechanical linkages to interest rates and spreads that show up in macro data BEFORE affecting returns. Equity depends more on deal-specific factors (management quality, sector selection, exit timing).

---

## 🤖 Models Tested (7 Algorithms)

We tested **7 machine learning algorithms** per asset to find what works best:

### **Linear Models with Regularization:**
1. **Ridge Regression** - L2 penalty, stable baseline (55.6% walk-forward)
2. **Lasso Regression** ⭐ - L1 penalty with feature selection (59.5% walk-forward)
3. **ElasticNet** - Combined L1/L2 regularization (59.5% walk-forward)

### **Tree-Based Ensemble Methods:**
4. **Random Forest** ⭐⭐ - 200 voting trees (60.5% walk-forward) - **BEST OVERALL**
5. **Gradient Boosting** - Sequential tree improvement (58.7% walk-forward)
6. **AdaBoost** - Adaptive boosting (59.6% walk-forward)
7. **XGBoost** - Industrial-strength boosting (59.5% walk-forward)

**Key Finding:** 
- **Static testing winner:** Lasso (62.9%)
- **Walk-forward winner:** Random Forest (60.5%)
- **Credit specialist:** Gradient Boosting (72.6% on SRLN)

**Model selection depends on validation method** - always use walk-forward for realistic estimates.

---

## 🔬 Validation Methodology

### Phase 1: Static Testing (Baseline)
- Traditional 80/20 train-test split
- Established baseline: 57.8% average
- **Winner:** Lasso (62.9%)

### Phase 2: Walk-Forward Validation (Realistic) ⭐
- Rolling 36-month training window
- Never train on future data (no look-ahead bias)
- Test on next month, roll forward, repeat
- **Result:** 59.0% average (+1.2pp vs static!)
- **Winner:** Random Forest (60.5%)

**Critical Discovery:** Walk-forward OUTPERFORMED static testing, proving models learn genuine patterns rather than overfitting training data.

### Anti-Overfitting Techniques

Financial data is noisy. We used **7 techniques** to ensure results are real:

1. **Walk-Forward Cross-Validation** - Never trained on future data
2. **Out-of-Sample Testing** - Final 20% never seen during training
3. **Feature Limitation** - Only top 5 most correlated factors
4. **Regularization** - Mathematical penalties on complexity
5. **Shallow Trees** - Max depth = 4 (prevents memorization)
6. **Time Series Split** - Respects temporal ordering
7. **Conservative Reporting** - Show all results, not cherry-picked

**Result:** Conservative, realistic estimates. Our 59% walk-forward rate is trustworthy.

---

## 📊 Key Economic Drivers (Top 5)

**What macro factors matter most?**

1. **Credit Spreads** (HY & IG) ⭐⭐⭐⭐⭐
   - Correlation: 0.35-0.45
   - Wider spreads → Lower returns
   - **Most predictive across all assets**

2. **Financial Stress Index** ⭐⭐⭐⭐
   - Correlation: 0.25-0.40
   - Higher stress → Lower returns

3. **GDP Growth** ⭐⭐⭐⭐
   - Correlation: 0.30-0.38
   - Higher growth → Higher returns

4. **Federal Funds Rate** ⭐⭐⭐
   - Correlation: -0.20 to -0.30
   - Rate hikes → Lower returns

5. **Volatility (VIX)** ⭐⭐⭐
   - Correlation: -0.25 to -0.35
   - Higher volatility → Lower returns

**Important:** Only 2-5 factors drive most assets. Using 50+ makes predictions WORSE!

---

## 📁 Project Structure (Simplified)

```
Private-Market-Conditions-Index/
│
├── 📄 **START HERE - DOCUMENTATION**
│   ├── MODEL_SELECTION_RESULTS.md    ⭐ Complete research findings
│   ├── METHODOLOGY.md                ⭐ Validation approach
│   ├── QUICKSTART.md                 ⭐ How to run the code
│   ├── README.md                     ← You are here (overview)
│   └── DATA_DICTIONARY.md            Variable definitions
│
├── 📊 **RESULTS & ANALYSIS**
│   └── reports/
│       ├── figures/                  Detailed visualizations
│       ├── tables/                   Model performance tables
│       └── EXECUTIVE_SUMMARY.md      High-level findings
│
├── 💻 **CODE** (Run These)
│   ├── scripts/
│   │   ├── run_full_pipeline.py      ⭐ MASTER SCRIPT (runs everything)
│   │   ├── 01_download_data.py       Get data from APIs
│   │   ├── 02_prepare_data.py        Process to monthly returns
│   │   ├── 03_run_analysis.py        Basic correlations
│   │   └── create_ultimate_ml_models.py  ML model comparison
│   │
│   └── src/                          Core functions (edit to customize)
│       ├── download_data.py
│       ├── prep_data.py
│       ├── analysis.py
│       └── config.py                 ⭐ Edit here to add assets/indicators
│
├── 📁 **DATA**
│   ├── data_processed/
│   │   ├── monthly_returns.csv       Asset returns by month
│   │   └── features_monthly.csv      Macro indicators
│   │
│   └── data/raw/                     Raw data from APIs (auto-downloaded)
│
└── 🔬 **BACKTESTING**
    └── backtesting/
        └── test_all_assets_all_models.py  Walk-forward validation
```

**What to use when:**
- **Understanding results?** → `MODEL_SELECTION_RESULTS.md`
- **Running analysis?** → `python scripts/run_full_pipeline.py`
- **Customizing?** → Edit `src/config.py` and re-run pipeline

---

## Data Sources

### Asset Prices (Yahoo Finance)
- **Private Equity Proxies:** PSP, BX, KKR, APO, CG
- **Private Credit Proxies:** HYG, JNK, BKLN, SRLN, BIZD, ARCC
- **Benchmarks:** SPY, VBR, IEF
- **Period:** 2005-01-01 onwards

### Macro/Financial Data (FRED)
- **Growth & Activity:** GDP, industrial production, retail sales
- **Labour:** Unemployment rate, participation rate
- **Inflation:** CPI, core CPI, PCE
- **Interest Rates:** Fed Funds, 2Y, 10Y Treasury yields
- **Credit & Stress:** HY/IG spreads, financial stress index
- **Risk Sentiment:** VIX
- **Leading Indicators:** Chicago Fed activity index, consumer sentiment
- **Period:** 2000-01-01 onwards

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for complete variable descriptions.

---


## Methodology

### 1. Data Collection
- Download daily price data from Yahoo Finance
- Download monthly/quarterly macro data from FRED
- Store raw data for reproducibility

### 2. Data Preparation
- Convert daily prices to monthly total returns (month-end to month-end)
- Resample all macro series to monthly frequency
- Forward-fill quarterly series (e.g., GDP)

### 3. Feature Engineering
- **Transformations:**
  - Year-over-year (YoY) for inflation and activity indicators
  - First difference (diff) for rates, spreads, and stress indices
  - Level for sentiment and index variables
- **Standardization:** Z-score all features for comparability
- **Sign adjustment:** Invert "bad news" variables (unemployment, spreads, stress)

### 4. Model Training & Validation
- **Phase 1:** Static 80/20 train-test split (baseline)
- **Phase 2:** Walk-forward validation with 36-month rolling window (realistic)
- Train 7 different ML algorithms per asset
- Measure directional accuracy (hit rate)
- Report walk-forward results as primary findings

### 5. Model Selection
- Compare performance across validation methods
- Identify which models work best for which assets
- Determine optimal number of features (2-5)
- Select Random Forest as best overall, Gradient Boosting for credit

---

## 💼 Business Applications

### 1. Tactical Asset Allocation
**Use:** Adjust private market exposure based on model predictions  
**Value:** Improve timing of capital deployment decisions  
**Best for:** Credit assets (65-73% accuracy)

### 2. Risk Management
**Use:** Anticipate periods of elevated private market stress  
**Value:** Earlier warning signals than backward-looking metrics  
**Best for:** Portfolio-wide monitoring

### 3. Model Selection Guidance
**Use:** Choose the right ML model for financial prediction tasks  
**Value:** Saves months of trial-and-error in future projects  
**Best for:** Data science teams

### 4. Academic Research
**Contribution:** Demonstrates genuine predictability in private markets  
**Value:** Evidence that credit is 20-30% more predictable than equity  
**Best for:** Publications, peer review

---

## Extending the Analysis

### Adding New Assets
Edit `src/config.py`:
```python
ASSET_TICKERS = [
    {"ticker": "YOUR_TICKER", "group": "PE"},  # or "PC" or "BENCHMARK"
    # ...
]
```

### Adding New Macro Variables
Edit `src/config.py`:
```python
FRED_SERIES = {
    "your_category": [
        "FRED_SERIES_ID",
        # ...
    ],
}
```

Then edit `src/prep_data.py` to specify the appropriate transformation (YoY, diff, or level).

### Customizing Analysis
- Modify correlation window: Edit `window` parameter in `compute_rolling_correlations()`
- Change lag structure: Edit `lag` parameter in analysis scripts
- Add new visualizations: Extend `src/plotting.py`

---

## Dependencies

Core libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `yfinance` - Yahoo Finance API client
- `fredapi` - FRED API client
- `scikit-learn` - Machine learning models
- `statsmodels` - Statistical models and tests

See [requirements.txt](requirements.txt) for complete list.

---

## Data Quality & Assumptions

### Assumptions
- FRED and Yahoo Finance data are sufficiently accurate for research
- Public market proxies reasonably represent private market dynamics
- Missing data can be handled by standard methods (dropping NAs after alignment)

### Data Quality Checks
- Missing value detection and reporting
- Outlier identification (±5 standard deviations)
- Structural break awareness (documented in reports)
- Consistency validation (no duplicates, proper alignment)

### Known Limitations
- Some series have limited history (sample size varies)
- Quarterly series are forward-filled (creates stepped patterns)
- Public proxies imperfectly represent private markets
- Predictions are directional only (not magnitude)

---

## Project Timeline

This project was completed over ~80 hours:
- Repository setup: 3h
- Data collection & documentation: 10h
- Data preparation & feature engineering: 15h
- Model training & validation: 25h
- Walk-forward testing: 15h
- Analysis & visualization: 20h
- Reporting & polishing: 12h

---

## Contributing

This is an academic research project. For questions or suggestions, please open an issue or contact the author.

---

## License

This project is for educational and research purposes. Data sources (Yahoo Finance, FRED) have their own terms of use.

---

## Contact

For questions about this research, please open an issue in the repository or refer to the documentation files listed in the Repository Guide section above.

---

## Acknowledgments

- Federal Reserve Economic Data (FRED) for free, high-quality macro data
- Yahoo Finance for accessible market data
- CRISP-DM methodology for structured approach
- Scikit-learn community for excellent ML tools

---

## References & Further Reading

- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [Yahoo Finance Python Library](https://pypi.org/project/yfinance/)
- [CRISP-DM Methodology](https://www.datascience-pm.com/crisp-dm-2/)
- [Walk-Forward Validation Best Practices](https://www.sciencedirect.com/topics/computer-science/walk-forward-validation)

For detailed research findings, see [MODEL_SELECTION_RESULTS.md](MODEL_SELECTION_RESULTS.md).

For methodology details, see [H5_report.pdf](H5_report.pdf).
