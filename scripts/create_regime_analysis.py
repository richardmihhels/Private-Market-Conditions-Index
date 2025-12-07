#!/usr/bin/env python3
"""
Regime-Based Correlation Analysis
Shows how macro-market relationships differ in good vs bad times.

Usage:
    python scripts/create_regime_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src import analysis

# Configure plotting
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# Human-readable labels
FEATURE_LABELS = {
    'a191rl1q225sbea_level_z': 'GDP Growth',
    'bamlh0a0hym2_diff_z': 'HY Spreads ⬇',
    'bamlc0a0cm_diff_z': 'IG Spreads ⬇',
    'cfnai_level_z': 'Economic Activity',
    'cpiaucsl_yoy_z': 'Inflation (CPI)',
    'cpilfesl_yoy_z': 'Core Inflation',
    'dgs10_diff_z': 'Long Rates',
    'dgs2_diff_z': 'Short Rates',
    'fedfunds_diff_z': 'Fed Rate Δ',
    'indpro_yoy_z': 'Industrial Prod',
    'stlfsi4_diff_z': 'Financial Stress ⬇',
    'umcsent_level_z': 'Consumer Conf',
    'unrate_diff_z': 'Unemployment ⬇',
    'vixcls_diff_z': 'Volatility ⬇',
    'yield_curve_slope_diff_z': 'Yield Curve ⬇',
    'rrsfs_yoy_z': 'Retail Sales',
    'dtwexbgs_level_z': 'Dollar',
    'pcepi_yoy_z': 'Inflation (PCE)',
    'civpart_diff_z': 'Labor Part',
}

ASSET_LABELS = {
    'PSP': 'PSP', 'BX': 'BX', 'KKR': 'KKR', 'APO': 'APO', 'CG': 'CG', 'VBR': 'VBR',
    'HYG': 'HYG', 'JNK': 'JNK', 'BKLN': 'BKLN', 'SRLN': 'SRLN', 'BIZD': 'BIZD', 'ARCC': 'ARCC',
}


def identify_regimes(returns_df, lookback_years=10):
    """Identify good vs bad months based on broad market performance."""
    # Use SPY as market benchmark
    spy_returns = returns_df[returns_df['ticker'] == 'SPY'].copy()
    spy_returns['date'] = pd.to_datetime(spy_returns['date'])
    spy_returns = spy_returns.sort_values('date')
    
    # Filter to last N years
    cutoff_date = spy_returns['date'].max() - pd.DateOffset(years=lookback_years)
    spy_returns = spy_returns[spy_returns['date'] >= cutoff_date]
    
    # Define regimes based on SPY returns
    median_return = spy_returns['return'].median()
    
    good_months = spy_returns[spy_returns['return'] > median_return]['date'].tolist()
    bad_months = spy_returns[spy_returns['return'] <= median_return]['date'].tolist()
    
    return good_months, bad_months, spy_returns


def compute_regime_correlations(returns_df, features_df, good_months, bad_months, asset_group):
    """Compute correlations separately for good and bad regimes."""
    returns_df = returns_df.copy()
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    
    # Filter by asset group
    group_returns = returns_df[returns_df['group'] == asset_group].copy()
    
    # Split into regimes
    good_returns = group_returns[group_returns['date'].isin(good_months)]
    bad_returns = group_returns[group_returns['date'].isin(bad_months)]
    
    # Compute correlations for each regime
    good_corr = analysis.compute_static_correlations(good_returns, features_df, lag=1)
    bad_corr = analysis.compute_static_correlations(bad_returns, features_df, lag=1)
    
    # Add labels
    for df in [good_corr, bad_corr]:
        df['feature_label'] = df['feature'].map(FEATURE_LABELS)
        df['asset_label'] = df['asset'].map(ASSET_LABELS)
    
    good_corr = good_corr.dropna(subset=['feature_label', 'asset_label'])
    bad_corr = bad_corr.dropna(subset=['feature_label', 'asset_label'])
    
    return good_corr, bad_corr


def create_regime_comparison_heatmap(good_corr, bad_corr, asset_group, output_dir):
    """Create side-by-side heatmaps for good vs bad regimes."""
    # Create pivots
    pivot_good = good_corr.pivot(index='feature_label', columns='asset_label', values='pearson')
    pivot_bad = bad_corr.pivot(index='feature_label', columns='asset_label', values='pearson')
    
    # Sort by average importance
    row_order = pivot_good.abs().mean(axis=1).add(pivot_bad.abs().mean(axis=1)).sort_values(ascending=False).index
    pivot_good = pivot_good.loc[row_order]
    pivot_bad = pivot_bad.loc[row_order]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Good months heatmap
    sns.heatmap(pivot_good, annot=True, fmt=".2f", cmap="RdYlGn", center=0, 
                vmin=-0.8, vmax=0.8, linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Correlation', 'shrink': 0.85},
                annot_kws={'size': 9, 'weight': 'bold'}, ax=ax1)
    
    ax1.set_title('📈 GOOD MONTHS (Market Up)\n' + 
                  f'{asset_group} Correlations When SPY > Median',
                  fontsize=15, fontweight='bold', pad=15, color='green')
    ax1.set_xlabel('\nAsset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Economic Factor\n', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=40, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)
    
    # Bad months heatmap
    sns.heatmap(pivot_bad, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-0.8, vmax=0.8, linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Correlation', 'shrink': 0.85},
                annot_kws={'size': 9, 'weight': 'bold'}, ax=ax2)
    
    ax2.set_title('📉 BAD MONTHS (Market Down)\n' + 
                  f'{asset_group} Correlations When SPY < Median',
                  fontsize=15, fontweight='bold', pad=15, color='red')
    ax2.set_xlabel('\nAsset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Economic Factor\n', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=40, labelsize=10)
    ax2.tick_params(axis='y', rotation=0, labelsize=10)
    
    plt.tight_layout()
    filename = f'{asset_group.lower().replace(" ", "_")}_regime_comparison.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_difference_heatmap(good_corr, bad_corr, asset_group, output_dir):
    """Create heatmap showing the DIFFERENCE between good and bad regimes."""
    pivot_good = good_corr.pivot(index='feature_label', columns='asset_label', values='pearson')
    pivot_bad = bad_corr.pivot(index='feature_label', columns='asset_label', values='pearson')
    
    # Calculate difference (Good - Bad)
    pivot_diff = pivot_good - pivot_bad
    
    # Sort by largest differences
    row_order = pivot_diff.abs().mean(axis=1).sort_values(ascending=False).index
    pivot_diff = pivot_diff.loc[row_order]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_diff, annot=True, fmt=".2f", cmap="PuOr", center=0,
                vmin=-0.6, vmax=0.6, linewidths=1.5, linecolor='white',
                cbar_kws={'label': 'Correlation Difference', 'shrink': 0.85},
                annot_kws={'size': 10, 'weight': 'bold'})
    
    plt.title(f'🔄 {asset_group}: Regime Sensitivity\n' +
              'How Correlations CHANGE in Good vs Bad Markets\n' +
              '(Positive = Stronger in Good Times | Negative = Stronger in Bad Times)',
              fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('\nAsset', fontsize=13, fontweight='bold')
    plt.ylabel('Economic Factor\n', fontsize=13, fontweight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Add legend box
    legend_text = ('PURPLE = More important in GOOD markets\n'
                   'ORANGE = More important in BAD markets\n'
                   'WHITE = Similar in both regimes')
    plt.text(0.02, 0.98, legend_text, transform=plt.gcf().transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    filename = f'{asset_group.lower().replace(" ", "_")}_regime_difference.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def create_summary_report(good_corr, bad_corr, good_months, bad_months, spy_returns, output_dir):
    """Create text summary of regime analysis."""
    lines = ["=" * 90]
    lines.append("REGIME-BASED CORRELATION ANALYSIS")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"Analysis Period: Last 10 Years ({spy_returns['date'].min().strftime('%Y-%m-%d')} to {spy_returns['date'].max().strftime('%Y-%m-%d')})")
    lines.append(f"Total Months Analyzed: {len(spy_returns)}")
    lines.append("")
    lines.append(f"📈 GOOD MONTHS: {len(good_months)} months (SPY returns > median = {spy_returns['return'].median():.2%})")
    lines.append(f"   Average SPY return: {spy_returns[spy_returns['date'].isin(good_months)]['return'].mean():.2%}")
    lines.append("")
    lines.append(f"📉 BAD MONTHS: {len(bad_months)} months (SPY returns ≤ median)")
    lines.append(f"   Average SPY return: {spy_returns[spy_returns['date'].isin(bad_months)]['return'].mean():.2%}")
    lines.append("")
    lines.append("=" * 90)
    lines.append("KEY INSIGHTS: How Relationships Change")
    lines.append("=" * 90)
    lines.append("")
    
    # Combine and find biggest differences
    good_corr['regime'] = 'good'
    bad_corr['regime'] = 'bad'
    combined = pd.concat([good_corr, bad_corr])
    
    # For each asset-feature pair, calculate difference
    differences = []
    for (asset, feature), group in combined.groupby(['asset_label', 'feature_label']):
        if len(group) == 2:
            good_val = group[group['regime'] == 'good']['pearson'].values[0]
            bad_val = group[group['regime'] == 'bad']['pearson'].values[0]
            diff = good_val - bad_val
            differences.append({
                'asset': asset,
                'feature': feature,
                'good_corr': good_val,
                'bad_corr': bad_val,
                'difference': diff,
                'abs_diff': abs(diff)
            })
    
    diff_df = pd.DataFrame(differences).sort_values('abs_diff', ascending=False)
    
    lines.append("🔥 TOP 10 BIGGEST REGIME DIFFERENCES\n")
    
    for i, row in diff_df.head(10).iterrows():
        if row['difference'] > 0:
            lines.append(f"📈 MORE IMPORTANT IN GOOD TIMES (+{row['difference']:.3f} difference)")
            lines.append(f"   {row['asset']} ↔ {row['feature']}")
            lines.append(f"   Good: {row['good_corr']:+.3f} | Bad: {row['bad_corr']:+.3f}")
            lines.append(f"   💡 This relationship is STRONGER when markets are rising")
        else:
            lines.append(f"📉 MORE IMPORTANT IN BAD TIMES ({row['difference']:.3f} difference)")
            lines.append(f"   {row['asset']} ↔ {row['feature']}")
            lines.append(f"   Good: {row['good_corr']:+.3f} | Bad: {row['bad_corr']:+.3f}")
            lines.append(f"   💡 This relationship is STRONGER when markets are falling")
        lines.append("")
    
    lines.append("=" * 90)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 90)
    lines.append("")
    lines.append("• Positive difference = Factor matters MORE in good times")
    lines.append("• Negative difference = Factor matters MORE in bad times (crisis predictor)")
    lines.append("• Small difference = Relationship is STABLE across regimes")
    lines.append("")
    lines.append("Use this to understand which indicators are:")
    lines.append("  - Pro-cyclical boosters (work best when times are good)")
    lines.append("  - Crisis indicators (become important when markets turn)")
    lines.append("  - All-weather indicators (consistent across both regimes)")
    
    # Save report
    with open(output_dir / 'regime_analysis_summary.txt', 'w') as f:
        f.write('\n'.join(lines))
    
    return 'regime_analysis_summary.txt'


def main():
    print("=" * 80)
    print("REGIME-BASED CORRELATION ANALYSIS")
    print("Good Times vs Bad Times (Last 10 Years)")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", index_col=0, parse_dates=True)
    
    print(f"  ✓ Loaded {len(returns):,} return observations")
    
    # Identify regimes
    print("\n🔍 Identifying market regimes (last 10 years)...")
    good_months, bad_months, spy_returns = identify_regimes(returns, lookback_years=10)
    
    print(f"  ✓ Good months (SPY > median): {len(good_months)}")
    print(f"  ✓ Bad months (SPY ≤ median): {len(bad_months)}")
    print(f"  ✓ Period: {spy_returns['date'].min().strftime('%Y-%m')} to {spy_returns['date'].max().strftime('%Y-%m')}")
    
    # Create output directory
    output_dir = Path("reports/figures_friendly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Analyze Private Equity
    print("\n📈 Analyzing Private Equity regimes...")
    pe_good, pe_bad = compute_regime_correlations(returns, features, good_months, bad_months, 'PE')
    
    print("  🎨 Creating PE comparison heatmap...")
    file1 = create_regime_comparison_heatmap(pe_good, pe_bad, 'Private Equity', output_dir)
    generated_files.append(file1)
    print(f"     ✓ Saved {file1}")
    
    print("  🎨 Creating PE difference heatmap...")
    file2 = create_difference_heatmap(pe_good, pe_bad, 'Private Equity', output_dir)
    generated_files.append(file2)
    print(f"     ✓ Saved {file2}")
    
    # Analyze Private Credit
    print("\n💰 Analyzing Private Credit regimes...")
    pc_good, pc_bad = compute_regime_correlations(returns, features, good_months, bad_months, 'PC')
    
    print("  🎨 Creating PC comparison heatmap...")
    file3 = create_regime_comparison_heatmap(pc_good, pc_bad, 'Private Credit', output_dir)
    generated_files.append(file3)
    print(f"     ✓ Saved {file3}")
    
    print("  🎨 Creating PC difference heatmap...")
    file4 = create_difference_heatmap(pc_good, pc_bad, 'Private Credit', output_dir)
    generated_files.append(file4)
    print(f"     ✓ Saved {file4}")
    
    # Create summary report
    print("\n📝 Creating summary report...")
    all_good = pd.concat([pe_good, pc_good])
    all_bad = pd.concat([pe_bad, pc_bad])
    file5 = create_summary_report(all_good, all_bad, good_months, bad_months, spy_returns, output_dir)
    generated_files.append(file5)
    print(f"  ✓ Saved {file5}")
    
    print("\n" + "=" * 80)
    print("✅ REGIME ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output location: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    for i, f in enumerate(generated_files, 1):
        print(f"  {i}. {f}")
    
    print("\n💡 KEY INSIGHT:")
    print("   Side-by-side heatmaps show how macro factors matter differently")
    print("   in good vs bad markets. Difference maps highlight regime-dependent factors!")
    print("\n   Look for:")
    print("   • Factors that flip from positive to negative (regime changers)")
    print("   • Factors that stay consistent (all-weather indicators)")
    print("   • Factors that only matter in crises (risk indicators)")


if __name__ == "__main__":
    main()



