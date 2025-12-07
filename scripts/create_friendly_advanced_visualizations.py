#!/usr/bin/env python3
"""
Create User-Friendly Advanced Regression Visualizations
Makes complex regression results easy to understand

Usage:
    python scripts/create_friendly_advanced_visualizations.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure plotting
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

FEATURE_LABELS = {
    'a191rl1q225sbea_level_z': 'GDP Growth',
    'bamlh0a0hym2_diff_z': 'HY Spreads ⬇',
    'bamlc0a0cm_diff_z': 'IG Spreads ⬇',
    'cfnai_level_z': 'Econ Activity',
    'cpiaucsl_yoy_z': 'Inflation',
    'fedfunds_diff_z': 'Fed Rate',
    'stlfsi4_diff_z': 'Fin Stress ⬇',
    'vixcls_diff_z': 'Volatility ⬇',
    'indpro_yoy_z': 'Production',
    'unrate_diff_z': 'Unemployment ⬇',
}

ASSET_LABELS = {
    'PSP': 'PSP', 'BX': 'Blackstone', 'KKR': 'KKR', 'APO': 'Apollo', 
    'CG': 'Carlyle', 'VBR': 'Small-Cap',
    'HYG': 'HY Bonds', 'JNK': 'Junk Bonds', 'BKLN': 'Bank Loans', 
    'SRLN': 'Senior Loans', 'BIZD': 'BDC ETF', 'ARCC': 'Ares Cap',
    'SPY': 'S&P 500', 'IEF': 'Treasuries'
}


def prepare_data(returns_df, features_df, lag=1):
    """Prepare aligned dataset."""
    returns_df = returns_df.copy()
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_wide = returns_df.pivot_table(index='date', columns='ticker', values='return')
    
    features_df = features_df.copy()
    features_df.index = pd.to_datetime(features_df.index)
    
    if lag > 0:
        features_df = features_df.shift(lag)
    
    combined = returns_wide.join(features_df, how='inner')
    return combined


def run_all_models(combined_df, asset, top_n=5):
    """Run multiple models and return results."""
    y = combined_df[asset].dropna()
    X_full = combined_df[list(FEATURE_LABELS.keys())].loc[y.index]
    
    valid_idx = X_full.notna().all(axis=1) & y.notna()
    X_full = X_full[valid_idx]
    y = y[valid_idx]
    
    if len(y) < 50:
        return None
    
    # Select top features
    correlations = X_full.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index.tolist()
    X = X_full[top_features]
    
    # Split data
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Multiple Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_test)
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_test_ridge = ridge.predict(X_test)
    
    # Lasso
    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_test_lasso = lasso.predict(X_test)
    
    # Hit rates
    hit_lr = (np.sign(y_test) == np.sign(y_pred_test_lr)).mean()
    hit_ridge = (np.sign(y_test) == np.sign(y_pred_test_ridge)).mean()
    hit_lasso = (np.sign(y_test) == np.sign(y_pred_test_lasso)).mean()
    
    return {
        'features': top_features,
        'train_r2': r2_score(y_train, y_pred_train_lr),
        'test_r2_lr': r2_score(y_test, y_pred_test_lr),
        'test_r2_ridge': r2_score(y_test, y_pred_test_ridge),
        'test_r2_lasso': r2_score(y_test, y_pred_test_lasso),
        'hit_rate_lr': hit_lr,
        'hit_rate_ridge': hit_ridge,
        'hit_rate_lasso': hit_lasso,
        'mae_lr': mean_absolute_error(y_test, y_pred_test_lr),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'coefficients': dict(zip(top_features, lr.coef_)),
        'test_dates': y_test.index,
        'y_test': y_test.values,
        'y_pred_lr': y_pred_test_lr,
        'y_pred_ridge': y_pred_test_ridge,
        'y_pred_lasso': y_pred_test_lasso,
    }


def create_model_comparison_dashboard(all_results, output_dir):
    """Create comprehensive comparison of all models."""
    # Prepare data
    data = []
    for asset, result in all_results.items():
        if result:
            data.append({
                'Asset': ASSET_LABELS.get(asset, asset),
                'Train R²': result['train_r2'],
                'Test R² (OLS)': result['test_r2_lr'],
                'Test R² (Ridge)': result['test_r2_ridge'],
                'Test R² (Lasso)': result['test_r2_lasso'],
                'Hit Rate': result['hit_rate_lr'],
                'MAE': result['mae_lr']
            })
    
    df = pd.DataFrame(data).sort_values('Hit Rate', ascending=False)
    
    # Create figure with 4 panels
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Model Performance Comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(len(df))
    width = 0.2
    
    bars1 = ax1.bar(x - width*1.5, df['Train R²'], width, label='Train (In-Sample)', 
                    color='lightblue', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x - width*0.5, df['Test R² (OLS)'], width, label='Test: Linear Regression', 
                    color='steelblue', edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width*0.5, df['Test R² (Ridge)'], width, label='Test: Ridge', 
                    color='orange', edgecolor='black', linewidth=1.5)
    bars4 = ax1.bar(x + width*1.5, df['Test R² (Lasso)'], width, label='Test: Lasso', 
                    color='green', edgecolor='black', linewidth=1.5)
    
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Asset'], rotation=45, ha='right', fontsize=11)
    ax1.set_ylabel('R² (Explanatory Power)', fontsize=14, fontweight='bold')
    ax1.set_title('📊 Model Performance: Can We Predict Future Returns?\n' +
                  '(Higher = Better, Negative = Worse than predicting average)',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper right', framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(min(-0.2, df['Test R² (OLS)'].min() - 0.1), max(0.3, df['Train R²'].max() + 0.1))
    
    # Panel 2: Directional Accuracy (Hit Rate)
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors = ['green' if x > 0.6 else 'orange' if x > 0.55 else 'red' for x in df['Hit Rate']]
    bars = ax2.barh(range(len(df)), df['Hit Rate'], color=colors, alpha=0.7, 
                    edgecolor='black', linewidth=1.5)
    
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(df['Asset'], fontsize=11)
    ax2.set_xlabel('Hit Rate (Correct Direction %)', fontsize=13, fontweight='bold')
    ax2.set_title('🎯 Directional Prediction Accuracy\n(Getting the sign right)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax2.axvline(0.6, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Good (60%)')
    ax2.set_xlim(0.4, max(0.7, df['Hit Rate'].max() + 0.05))
    ax2.grid(axis='x', alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add value labels
    for i, val in enumerate(df['Hit Rate']):
        ax2.text(val, i, f'  {val:.1%}', va='center', fontsize=11, fontweight='bold')
    
    # Panel 3: Overfitting Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    
    df['Overfitting'] = df['Train R²'] - df['Test R² (OLS)']
    colors = ['green' if x < 0.1 else 'orange' if x < 0.3 else 'red' for x in df['Overfitting']]
    
    bars = ax3.barh(range(len(df)), df['Overfitting'], color=colors, alpha=0.7,
                    edgecolor='black', linewidth=1.5)
    
    ax3.set_yticks(range(len(df)))
    ax3.set_yticklabels(df['Asset'], fontsize=11)
    ax3.set_xlabel('Performance Drop (Train R² - Test R²)', fontsize=13, fontweight='bold')
    ax3.set_title('⚠️ Overfitting Check\n(Lower = More Reliable)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.axvline(0.1, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Good (<0.1)')
    ax3.axvline(0.3, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='Moderate')
    ax3.grid(axis='x', alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Add value labels
    for i, val in enumerate(df['Overfitting']):
        ax3.text(val, i, f'  {val:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # Add overall legend box
    legend_text = (
        '📖 HOW TO READ THIS:\n\n'
        'Top Panel: R² shows how much returns are explained\n'
        '  • Train (light blue) = How well we fit historical data\n'
        '  • Test (colors) = How well we predict NEW data\n'
        '  • Negative Test R² = Model fails to predict\n\n'
        'Bottom Left: Hit Rate = % of times we predict correct direction\n'
        '  • Green (>60%) = Good directional prediction\n'
        '  • Red (<55%) = No better than guessing\n\n'
        'Bottom Right: Overfitting = Performance drop from train to test\n'
        '  • Green (<0.1) = Stable, reliable\n'
        '  • Red (>0.3) = Memorizes past, fails future'
    )
    
    fig.text(0.02, 0.98, legend_text, transform=fig.transFigure,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             family='monospace')
    
    plt.savefig(output_dir / 'advanced_regression_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'advanced_regression_dashboard.png'


def create_best_vs_worst_forecast_plots(all_results, output_dir):
    """Show best and worst performing models side by side."""
    # Find best and worst by hit rate
    hit_rates = {asset: result['hit_rate_lr'] for asset, result in all_results.items() if result}
    best_asset = max(hit_rates, key=hit_rates.get)
    worst_asset = min(hit_rates, key=hit_rates.get)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, asset in enumerate([best_asset, worst_asset]):
        result = all_results[asset]
        ax = axes[idx]
        
        # Plot
        dates = result['test_dates']
        ax.plot(dates, result['y_test'], 'o-', label='Actual Returns', 
                color='blue', linewidth=2.5, markersize=8, alpha=0.8)
        ax.plot(dates, result['y_pred_lr'], 's--', label='Predicted Returns', 
                color='red', linewidth=2, markersize=7, alpha=0.8)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Monthly Return', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Title and stats
        asset_name = ASSET_LABELS.get(asset, asset)
        status = 'BEST' if idx == 0 else 'WORST'
        color = 'green' if idx == 0 else 'red'
        
        ax.set_title(f'{status} Model: {asset_name}\n' +
                     f'Hit Rate = {result["hit_rate_lr"]:.1%} | Test R² = {result["test_r2_lr"]:.3f}',
                     fontsize=14, fontweight='bold', color=color, pad=15)
        
        # Add stats box
        stats_text = (f'Train R²: {result["train_r2"]:.3f}\n'
                     f'Test R²: {result["test_r2_lr"]:.3f}\n'
                     f'Hit Rate: {result["hit_rate_lr"]:.1%}\n'
                     f'MAE: {result["mae_lr"]:.4f}\n'
                     f'Test Size: {result["n_test"]} months')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                family='monospace')
    
    plt.suptitle('📊 Best vs Worst Performing Forecast Models\n' +
                 'Out-of-Sample Test Period (Last 20% of Data)',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_vs_worst_forecasts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'best_vs_worst_forecasts.png'


def create_feature_importance_chart(all_results, output_dir):
    """Show which features matter most across all assets."""
    # Collect all features and their usage
    feature_counts = {}
    feature_avg_coefs = {}
    
    for asset, result in all_results.items():
        if result:
            for feature, coef in result['coefficients'].items():
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                    feature_avg_coefs[feature] = []
                feature_counts[feature] += 1
                feature_avg_coefs[feature].append(abs(coef))
    
    # Calculate average importance
    feature_importance = {feat: (feature_counts[feat], np.mean(feature_avg_coefs[feat])) 
                         for feat in feature_counts}
    
    # Create DataFrame
    imp_df = pd.DataFrame([
        {'Feature': FEATURE_LABELS.get(feat, feat), 
         'Usage Count': vals[0],
         'Avg |Coefficient|': vals[1]}
        for feat, vals in feature_importance.items()
    ]).sort_values('Usage Count', ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Panel 1: Feature Selection Frequency
    colors1 = plt.cm.RdYlGn(imp_df['Usage Count'] / imp_df['Usage Count'].max())
    bars1 = ax1.barh(range(len(imp_df)), imp_df['Usage Count'], color=colors1,
                     edgecolor='black', linewidth=1.5)
    ax1.set_yticks(range(len(imp_df)))
    ax1.set_yticklabels(imp_df['Feature'], fontsize=12)
    ax1.set_xlabel('Number of Assets Using This Factor', fontsize=14, fontweight='bold')
    ax1.set_title('🎯 Most Commonly Selected Factors\n(Top 5 features per asset)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    for i, val in enumerate(imp_df['Usage Count']):
        ax1.text(val, i, f'  {int(val)} assets', va='center', fontsize=11, fontweight='bold')
    
    # Panel 2: Average Effect Size
    colors2 = plt.cm.YlOrRd(imp_df['Avg |Coefficient|'] / imp_df['Avg |Coefficient|'].max())
    bars2 = ax2.barh(range(len(imp_df)), imp_df['Avg |Coefficient|'] * 100, color=colors2,
                     edgecolor='black', linewidth=1.5)
    ax2.set_yticks(range(len(imp_df)))
    ax2.set_yticklabels(imp_df['Feature'], fontsize=12)
    ax2.set_xlabel('Average Effect Size (% per std dev)', fontsize=14, fontweight='bold')
    ax2.set_title('💪 Strongest Average Impact\n(When selected)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, val in enumerate(imp_df['Avg |Coefficient|']):
        ax2.text(val * 100, i, f'  {val*100:.2f}%', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'feature_importance_summary.png'


def create_simple_summary_table(all_results, output_dir):
    """Create easy-to-read summary table."""
    data = []
    for asset, result in all_results.items():
        if result:
            verdict = ''
            if result['hit_rate_lr'] > 0.6 and result['test_r2_lr'] > 0:
                verdict = '✅ USEFUL'
            elif result['hit_rate_lr'] > 0.55:
                verdict = '~OK (Direction Only)'
            else:
                verdict = '❌ NOT RELIABLE'
            
            data.append({
                'Asset': ASSET_LABELS.get(asset, asset),
                'Hit_Rate': result['hit_rate_lr'],
                'Test_R2': result['test_r2_lr'],
                'Train_R2': result['train_r2'],
                'Verdict': verdict
            })
    
    df = pd.DataFrame(data).sort_values('Hit_Rate', ascending=False)
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Asset', 'Hit Rate', 'Test R²', 'Train R²', 'Verdict'])
    
    for _, row in df.iterrows():
        table_data.append([
            row['Asset'],
            f"{row['Hit_Rate']:.1%}",
            f"{row['Test_R2']:.3f}",
            f"{row['Train_R2']:.3f}",
            row['Verdict']
        ])
    
    # Color code the table
    colors = [['lightgray'] * 5]  # Header
    for _, row in df.iterrows():
        if '✅' in row['Verdict']:
            row_color = ['lightgreen'] * 5
        elif '~OK' in row['Verdict']:
            row_color = ['lightyellow'] * 5
        else:
            row_color = ['lightcoral'] * 5
        colors.append(row_color)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     cellColours=colors, bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Bold header
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold', fontsize=14)
    
    plt.title('📊 QUICK REFERENCE: Model Performance Summary\n' +
              '(Out-of-Sample Testing Results)',
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = (
        '✅ USEFUL = Hit Rate > 60% AND Test R² > 0 (Good for both direction and magnitude)\n'
        '~OK = Hit Rate > 55% (Good for direction, but poor magnitude prediction)\n'
        '❌ NOT RELIABLE = Hit Rate ≤ 55% (No better than guessing)\n\n'
        'Hit Rate = % of times model predicts correct direction (up/down)\n'
        'Test R² = How well model predicts magnitude on NEW data (negative = worse than average)'
    )
    
    fig.text(0.5, 0.05, legend_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.savefig(output_dir / 'model_performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'model_performance_table.png'


def main():
    print("=" * 80)
    print("CREATING USER-FRIENDLY ADVANCED REGRESSION VISUALIZATIONS")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", index_col=0, parse_dates=True)
    
    # Prepare
    print("📝 Preparing data and running models...")
    combined = prepare_data(returns, features, lag=1)
    
    # Run models for all assets
    all_results = {}
    assets = returns['ticker'].unique()
    
    for asset in assets:
        result = run_all_models(combined, asset, top_n=5)
        if result:
            all_results[asset] = result
    
    print(f"  ✓ Analyzed {len(all_results)} assets")
    
    # Create output directory
    output_dir = Path("reports/figures_friendly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Create visualizations
    print("\n🎨 Creating comprehensive dashboard...")
    file1 = create_model_comparison_dashboard(all_results, output_dir)
    generated_files.append(file1)
    print(f"  ✓ Saved {file1}")
    
    print("\n🎨 Creating best vs worst comparison...")
    file2 = create_best_vs_worst_forecast_plots(all_results, output_dir)
    generated_files.append(file2)
    print(f"  ✓ Saved {file2}")
    
    print("\n🎨 Creating feature importance chart...")
    file3 = create_feature_importance_chart(all_results, output_dir)
    generated_files.append(file3)
    print(f"  ✓ Saved {file3}")
    
    print("\n🎨 Creating summary table...")
    file4 = create_simple_summary_table(all_results, output_dir)
    generated_files.append(file4)
    print(f"  ✓ Saved {file4}")
    
    print("\n" + "=" * 80)
    print("✅ USER-FRIENDLY VISUALIZATIONS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output location: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    for i, f in enumerate(generated_files, 1):
        print(f"  {i}. {f}")
    
    print("\n💡 START HERE:")
    print("   1. model_performance_table.png - Quick summary (color-coded!)")
    print("   2. advanced_regression_dashboard.png - Complete overview")
    print("   3. best_vs_worst_forecasts.png - See what works vs what doesn't")
    print("   4. feature_importance_summary.png - Which factors matter most")
    
    print("\n🎯 These are designed for presentations and reports!")


if __name__ == "__main__":
    main()



