# Data Dictionary

## Overview
This document describes all variables used in the Private Market Conditions Index project, including asset tickers, macro/financial indicators, and their transformations.

---

## Asset Tickers

### Private Equity (PE) Proxies
| Ticker | Name | Description |
|--------|------|-------------|
| PSP | Invesco Global Listed Private Equity ETF | ETF tracking global listed private equity firms |
| BX | Blackstone Inc. | Leading alternative asset manager (PE, real estate, credit) |
| KKR | KKR & Co. Inc. | Global investment firm focused on private equity |
| APO | Apollo Global Management | Alternative investment manager (PE, credit, real estate) |
| CG | The Carlyle Group | Global investment firm in PE, credit, and investment solutions |
| VBR | Vanguard Small-Cap Value ETF | Small-cap value proxy for PE-like returns |

### Private Credit (PC) Proxies
| Ticker | Name | Description |
|--------|------|-------------|
| HYG | iShares iBoxx $ High Yield Corporate Bond ETF | High yield corporate bond exposure |
| JNK | SPDR Bloomberg High Yield Bond ETF | High yield bond market exposure |
| BKLN | Invesco Senior Loan ETF | Senior secured floating rate bank loan exposure |
| SRLN | SPDR Blackstone Senior Loan ETF | Senior secured loan exposure |
| BIZD | VanEck BDC Income ETF | Business Development Company (direct lender) exposure |
| ARCC | Ares Capital Corporation | Leading publicly traded business development company |

### Benchmarks
| Ticker | Name | Description |
|--------|------|-------------|
| SPY | SPDR S&P 500 ETF | Broad US equity market benchmark |
| IEF | iShares 7-10 Year Treasury Bond ETF | Intermediate-term US Treasury benchmark |

---

## FRED Macro/Financial Series

### Growth & Activity
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| A191RL1Q225SBEA | Real GDP Growth | % Change | Quarterly real GDP growth rate, annualized |
| INDPRO | Industrial Production Index | Index | Measure of real output in manufacturing, mining, utilities |
| PAYEMS | Total Nonfarm Payrolls | Thousands | Number of paid US workers (employment indicator) |
| RRSFS | Real Retail Sales | Millions of $ | Inflation-adjusted retail sales (consumer spending) |

### Labour Market
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| UNRATE | Unemployment Rate | % | Percentage of labor force that is unemployed |
| CIVPART | Civilian Participation Rate | % | Labor force as percentage of working-age population |

### Inflation
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| CPIAUCSL | Consumer Price Index (All Items) | Index | Headline inflation measure |
| CPILFESL | CPI Less Food & Energy (Core CPI) | Index | Core inflation excluding volatile food/energy |
| PCEPI | Personal Consumption Expenditure Price Index | Index | Fed's preferred inflation gauge |

### Interest Rates
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| FEDFUNDS | Federal Funds Rate | % | Target rate set by Federal Reserve (overnight lending) |
| DGS2 | 2-Year Treasury Yield | % | Market yield on 2-year US Treasury securities |
| DGS10 | 10-Year Treasury Yield | % | Market yield on 10-year US Treasury securities |

### Credit & Financial Stress
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| BAMLH0A0HYM2 | High Yield OAS | % | Option-adjusted spread on US high yield bonds vs Treasuries |
| BAMLC0A0CM | Investment Grade OAS | % | Option-adjusted spread on US investment grade bonds |
| STLFSI4 | St. Louis Fed Financial Stress Index | Index | Weekly measure of financial market stress (higher = more stress) |

### Risk Sentiment
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| VIXCLS | CBOE Volatility Index (VIX) | Index | Market expectation of 30-day volatility (fear gauge) |

### Leading Indicators & Sentiment
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| CFNAI | Chicago Fed National Activity Index | Index | Weighted average of 85 economic indicators (0 = trend growth) |
| UMCSENT | University of Michigan Consumer Sentiment | Index | Consumer confidence and expectations |

### Cross-Asset
| Series ID | Name | Unit | Description |
|-----------|------|------|-------------|
| DTWEXBGS | Trade Weighted US Dollar Index | Index | Broad measure of dollar strength vs trading partners |

### Derived Variables
| Variable | Calculation | Description |
|----------|-------------|-------------|
| YIELD_CURVE_SLOPE | DGS10 - DGS2 | Difference between 10Y and 2Y yields (yield curve steepness) |

---

## Feature Engineering

### Transformation Types

**1. Year-over-Year (YoY) Growth**
- Applied to: Price indices (CPI, PCE), activity indicators (INDPRO, RRSFS)
- Calculation: `(X_t / X_{t-12}) - 1`
- Captures annual growth rates
- Variables: `cpiaucsl_yoy`, `cpilfesl_yoy`, `pcepi_yoy`, `indpro_yoy`, `rrsfs_yoy`

**2. First Difference (Diff)**
- Applied to: Interest rates, spreads, stress indices, unemployment
- Calculation: `X_t - X_{t-1}`
- Captures monthly changes in levels
- Variables: `fedfunds_diff`, `dgs2_diff`, `dgs10_diff`, `bamlh0a0hym2_diff`, `bamlc0a0cm_diff`, `stlfsi4_diff`, `vixcls_diff`, `unrate_diff`, `yield_curve_slope_diff`

**3. Level**
- Applied to: Indices and ratios already in interpretable units
- Calculation: No transformation
- Variables: `cfnai_level`, `umcsent_level`, `dtwexbgs_level`

**4. Z-Score Standardization**
- Applied to: ALL features after transformation
- Calculation: `(X - mean(X)) / std(X)`
- Makes variables comparable by normalizing to standard deviations
- All features end with `_z` suffix (e.g., `cpiaucsl_yoy_z`)

### Sign Adjustments
For economic intuition, certain "bad news" variables are inverted so that **higher values = better conditions**:
- `UNRATE` (unemployment) → negated
- `BAMLH0A0HYM2` (HY spread) → negated (wider spreads = worse)
- `BAMLC0A0CM` (IG spread) → negated
- `STLFSI4` (financial stress) → negated
- `VIXCLS` (volatility) → negated
- `YIELD_CURVE_SLOPE` → negated (inverted curve = recession risk)

This ensures positive correlations mean "better macro → better returns" across all features.

---

## Data Frequency & Alignment

- **Raw price data**: Daily (from Yahoo Finance)
- **Monthly returns**: Calculated as month-end to month-end percentage change
- **FRED macro data**: Mix of monthly and quarterly
  - Quarterly series (GDP): Forward-filled to monthly frequency
  - Monthly series: Resampled to month-end
- **Feature matrix**: All features aligned to common monthly index

---

## Date Ranges

- **Price data**: 2005-01-01 onwards (captures GFC, COVID, recent cycles)
- **Macro data**: 2000-01-01 onwards (allows for lagged features and YoY calculations)
- **Analysis period**: Determined by intersection of all series after transformations

---

## Interpretation Guide

### Correlations with Asset Returns

**Positive Correlation (+)**
- Higher values of the feature → higher asset returns
- Example: Positive correlation with GDP growth means asset performs better in strong growth

**Negative Correlation (-)**
- Higher values of the feature → lower asset returns
- Example: Negative correlation with (non-inverted) HY spreads means wider credit spreads hurt returns

**Lag Interpretation**
- `lag=0`: Contemporaneous (same month)
- `lag=1`: Feature leads by 1 month (prior month's macro predicts current returns)

### Economic Interpretation

**Private Equity typically sensitive to:**
- Growth indicators (GDP, payrolls, industrial production)
- Credit spreads (tighter = easier financing)
- Risk sentiment (VIX, financial stress)

**Private Credit typically sensitive to:**
- Credit spreads (direct exposure to credit risk)
- Interest rate changes (floating rate exposure)
- Financial stress indicators
- Default risk proxies (unemployment, economic weakness)

---

## Data Sources

- **Price Data**: Yahoo Finance (`yfinance` Python library)
- **Macro Data**: Federal Reserve Economic Data (FRED) via API (`fredapi` Python library)

---

## Usage Notes

1. All monetary series are in nominal terms unless specified as "Real"
2. Index series (CPI, INDPRO) are typically used in growth rate form
3. Missing data is handled by dropping observations after alignment
4. Z-scores are calculated on the full sample (not rolling) for consistency




