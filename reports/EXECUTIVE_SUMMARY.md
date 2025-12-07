# Executive Summary: Private Markets & Economic Data Correlations

## Bottom Line

**Economic indicators show strong contemporaneous relationships with private market performance, but weak predictive power at one-month horizons.**

---

## Key Findings at a Glance

### ✅ Strong Relationships (Same Month)

| Indicator | PE Correlation | PC Correlation | Interpretation |
|-----------|----------------|----------------|----------------|
| **Credit Spreads (HY/IG)** | 0.58 - 0.79 | 0.74 - 0.88 | **STRONGEST DRIVER** |
| **Risk Sentiment (VIX)** | 0.58 - 0.70 | 0.59 - 0.66 | **STRONG COINCIDENT INDICATOR** |

### ⚠️ Weak Relationships

| Indicator | PE Correlation | PC Correlation | Interpretation |
|-----------|----------------|----------------|----------------|
| **Interest Rates** | 0.06 - 0.23 | 0.10 - 0.27 | Not primary drivers |
| **Inflation** | -0.09 - -0.18 | -0.10 - -0.23 | Weak headwind |
| **Growth Indicators** | 0.01 - 0.11 | -0.07 - 0.17 | Very weak |

### ❌ No Predictive Power (1-Month Lag)

- **All lag-1 correlations are weak** (|r| < 0.3)
- Economic indicators are **coincident, not predictive** at short horizons
- Credit spreads lose predictive power when lagged (r drops from 0.66 to 0.04)

---

## Top Correlations

### Private Equity (PE)
1. **PSP vs HY Spread**: r = 0.79 (same month)
2. **VBR vs HY Spread**: r = 0.74 (same month)
3. **VBR vs VIX**: r = 0.70 (same month)

### Private Credit (PC)
1. **JNK vs HY Spread**: r = 0.88 (same month)
2. **SRLN vs HY Spread**: r = 0.87 (same month)
3. **BKLN vs HY Spread**: r = 0.86 (same month)

---

## Practical Implications

### ✅ What Works
- **Monitor credit spreads in real-time** to assess current market conditions
- **Use VIX as a risk gauge** for current market sentiment
- **Credit spreads are stable indicators** (relationships persist over time)

### ❌ What Doesn't Work
- **Don't use economic indicators for 1-month forecasting** (weak predictive power)
- **Don't overweight interest rates** (weak relationships)
- **Don't rely on growth indicators** (very weak relationships)

### 💡 Recommendations
1. **Build real-time monitoring dashboard** for credit spreads and VIX
2. **Use as coincident indicators** (not predictive)
3. **Adjust allocation based on current spread levels** (wider = more attractive)
4. **Monitor correlation stability** over time

---

## Asset Class Differences

| Metric | Private Equity | Private Credit |
|--------|----------------|----------------|
| **Credit Spread Sensitivity** | Moderate (r = 0.58) | High (r = 0.74) |
| **Risk Sentiment Sensitivity** | High (r = 0.58) | Moderate (r = 0.59) |
| **Interest Rate Sensitivity** | Very Low (r = 0.06) | Low (r = 0.10) |

**Key Insight**: PC assets are more sensitive to credit conditions, while PE assets show similar sensitivity to risk sentiment.

---

## Data & Methodology

- **Period**: 2000-2024 (monthly data)
- **Assets**: PE proxies (PSP, BX, KKR, APO, CG, VBR) and PC proxies (HYG, JNK, BKLN, SRLN, BIZD, ARCC)
- **Indicators**: 15+ FRED series (growth, labour, inflation, rates, credit, risk)
- **Analysis**: Pearson and Spearman correlations, rolling correlations, category analysis

---

## Next Steps

1. **Test longer lags** (3-month, 6-month, 12-month) for predictive power
2. **Regime analysis** (recession vs expansion periods)
3. **Non-linear relationships** (threshold effects)
4. **Multi-factor models** combining multiple indicators
5. **Cross-asset relative value** analysis

---

*For detailed analysis, see `CORRELATION_ANALYSIS_REPORT.md`*



