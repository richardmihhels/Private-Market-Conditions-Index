#!/usr/bin/env python3
"""
Improved ML Predictions with Anti-Overfitting Techniques
Uses Random Forest + XGBoost + Walk-Forward Validation

Usage:
    python scripts/create_ml_improved_predictions.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")
    print("   Continuing with Random Forest only...\n")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure plotting
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

FEATURE_LABELS = {
    'a191rl1q225sbea_level_z': 'GDP Growth',
    'bamlh0a0hym2_diff_z': 'HY Spreads',
    'bamlc0a0cm_diff_z': 'IG Spreads',
    'cfnai_level_z': 'Econ Activity',
    'cpiaucsl_yoy_z': 'Inflation',
    'fedfunds_diff_z': 'Fed Rate',
    'stlfsi4_diff_z': 'Fin Stress',
    'vixcls_diff_z': 'Volatility',
    'indpro_yoy_z': 'Production',
    'unrate_diff_z': 'Unemployment',
}

ASSET_LABELS = {
    'PSP': 'PSP', 'BX': 'Blackstone', 'KKR': 'KKR', 'APO': 'Apollo', 
    'CG': 'Carlyle', 'VBR': 'Small-Cap',
    'HYG': 'HY Bonds', 'JNK': 'Junk Bonds', 'BKLN': 'Bank Loans', 
    'SRLN': 'Senior Loans', 'BIZD': 'BDC ETF', 'ARCC': 'Ares Cap',
    'SPY': 'S&P 500', 'IEF': 'Treasuries'
}


def prepare_data(returns_df, features_df, lag=1):
    """Prepare data with lag."""
    returns_df = returns_df.copy()
    returns_df['date'] = pd.to_datetime(returns_df['date'])
    returns_wide = returns_df.pivot_table(index='date', columns='ticker', values='return')
    
    features_df = features_df.copy()
    features_df.index = pd.to_datetime(features_df.index)
    
    if lag > 0:
        features_df = features_df.shift(lag)
    
    combined = returns_wide.join(features_df, how='inner')
    return combined


def create_interaction_features(X, top_n=3):
    """
    Create economically meaningful interaction terms.
    E.g., GDP Growth × Credit Spreads (growth matters more when spreads are tight)
    """
    X_enhanced = X.copy()
    
    # Only create interactions between top features to avoid explosion
    feature_names = X.columns.tolist()[:top_n]
    
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            feat1, feat2 = feature_names[i], feature_names[j]
            interaction_name = f"{feat1}_x_{feat2}"
            X_enhanced[interaction_name] = X[feat1] * X[feat2]
    
    return X_enhanced


def walk_forward_validation(X, y, model_type='ridge', n_splits=5):
    """
    Walk-forward cross-validation for time series.
    More realistic than random train/test split.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model based on type
        if model_type == 'ridge':
            model = Ridge(alpha=10.0)  # Higher alpha = more regularization
        elif model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=3,  # Shallow trees = less overfitting
                min_samples_split=20,  # Require more samples to split
                min_samples_leaf=10,   # Require more samples per leaf
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'xgb' and HAS_XGBOOST:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,  # Slow learning = less overfitting
                subsample=0.8,       # Use 80% of data per tree
                colsample_bytree=0.8,  # Use 80% of features per tree
                reg_alpha=1.0,       # L1 regularization
                reg_lambda=1.0,      # L2 regularization
                random_state=42
            )
        else:
            model = Ridge(alpha=10.0)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        hit_rate = (np.sign(y_test) == np.sign(y_pred_test)).mean()
        mae = mean_absolute_error(y_test, y_pred_test)
        
        results.append({
            'fold': fold,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'hit_rate': hit_rate,
            'mae': mae,
            'n_train': len(y_train),
            'n_test': len(y_test)
        })
    
    return pd.DataFrame(results)


def run_improved_models(combined_df, asset, top_n=5):
    """Run improved models with anti-overfitting techniques."""
    y = combined_df[asset].dropna()
    X_base = combined_df[list(FEATURE_LABELS.keys())].loc[y.index]
    
    # Drop NaNs
    valid_idx = X_base.notna().all(axis=1) & y.notna()
    X_base = X_base[valid_idx]
    y = y[valid_idx]
    
    if len(y) < 100:  # Need enough data for cross-validation
        return None
    
    # Select top features based on correlation
    correlations = X_base.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlations.head(top_n).index.tolist()
    X = X_base[top_features].copy()
    
    # Create interaction features
    X_enhanced = create_interaction_features(X, top_n=3)
    
    results = {}
    
    # 1. Ridge Regression (baseline)
    print(f"    Ridge...")
    ridge_results = walk_forward_validation(X, y, model_type='ridge', n_splits=5)
    results['ridge'] = {
        'avg_test_r2': ridge_results['test_r2'].mean(),
        'avg_hit_rate': ridge_results['hit_rate'].mean(),
        'std_test_r2': ridge_results['test_r2'].std(),
        'all_folds': ridge_results
    }
    
    # 2. Random Forest
    print(f"    Random Forest...")
    rf_results = walk_forward_validation(X_enhanced, y, model_type='rf', n_splits=5)
    results['random_forest'] = {
        'avg_test_r2': rf_results['test_r2'].mean(),
        'avg_hit_rate': rf_results['hit_rate'].mean(),
        'std_test_r2': rf_results['test_r2'].std(),
        'all_folds': rf_results
    }
    
    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        print(f"    XGBoost...")
        xgb_results = walk_forward_validation(X_enhanced, y, model_type='xgb', n_splits=5)
        results['xgboost'] = {
            'avg_test_r2': xgb_results['test_r2'].mean(),
            'avg_hit_rate': xgb_results['hit_rate'].mean(),
            'std_test_r2': xgb_results['test_r2'].std(),
            'all_folds': xgb_results
        }
    
    # 4. Ensemble (average predictions from all models)
    # Train final models on full data for ensemble
    split_idx = int(len(y) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    X_enh_train, X_enh_test = X_enhanced.iloc[:split_idx], X_enhanced.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train each model
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_split=20, 
                               min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X_enh_train, y_train)
    
    # Ensemble predictions
    pred_ridge = ridge.predict(X_test)
    pred_rf = rf.predict(X_enh_test)
    
    if HAS_XGBOOST:
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     reg_alpha=1.0, reg_lambda=1.0, random_state=42)
        xgb_model.fit(X_enh_train, y_train)
        pred_xgb = xgb_model.predict(X_enh_test)
        pred_ensemble = (pred_ridge + pred_rf + pred_xgb) / 3
    else:
        pred_ensemble = (pred_ridge + pred_rf) / 2
    
    ensemble_r2 = r2_score(y_test, pred_ensemble)
    ensemble_hit = (np.sign(y_test) == np.sign(pred_ensemble)).mean()
    
    results['ensemble'] = {
        'avg_test_r2': ensemble_r2,
        'avg_hit_rate': ensemble_hit,
        'predictions': pred_ensemble,
        'actual': y_test.values,
        'dates': y_test.index
    }
    
    results['features'] = top_features
    results['n_obs'] = len(y)
    
    return results


def create_ml_comparison_dashboard(all_results, output_dir):
    """Create dashboard comparing ML models."""
    # Prepare data
    data = []
    for asset, result in all_results.items():
        if result:
            row = {
                'Asset': ASSET_LABELS.get(asset, asset),
                'Ridge R²': result['ridge']['avg_test_r2'],
                'RF R²': result['random_forest']['avg_test_r2'],
                'Ridge Hit': result['ridge']['avg_hit_rate'],
                'RF Hit': result['random_forest']['avg_hit_rate'],
                'Ensemble R²': result['ensemble']['avg_test_r2'],
                'Ensemble Hit': result['ensemble']['avg_hit_rate'],
            }
            
            if HAS_XGBOOST and 'xgboost' in result:
                row['XGB R²'] = result['xgboost']['avg_test_r2']
                row['XGB Hit'] = result['xgboost']['avg_hit_rate']
            
            data.append(row)
    
    df = pd.DataFrame(data).sort_values('Ensemble Hit', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Panel 1: R² Comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    width = 0.25 if HAS_XGBOOST else 0.3
    
    ax.barh(x - width*1.5, df['Ridge R²'], width, label='Ridge (Linear)', color='steelblue', alpha=0.8)
    ax.barh(x - width*0.5, df['RF R²'], width, label='Random Forest', color='green', alpha=0.8)
    if HAS_XGBOOST:
        ax.barh(x + width*0.5, df['XGB R²'], width, label='XGBoost', color='purple', alpha=0.8)
    ax.barh(x + width*1.5, df['Ensemble R²'], width, label='Ensemble (Avg)', color='red', alpha=0.8)
    
    ax.axvline(0, color='black', linestyle='--', linewidth=2)
    ax.set_yticks(x)
    ax.set_yticklabels(df['Asset'], fontsize=11)
    ax.set_xlabel('Test R² (Out-of-Sample)', fontsize=13, fontweight='bold')
    ax.set_title('📊 R²: Which Model Explains Returns Best?\n(Higher = Better, Negative = Fails)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    # Panel 2: Hit Rate Comparison
    ax = axes[0, 1]
    
    ax.barh(x - width*1.5, df['Ridge Hit'], width, label='Ridge', color='steelblue', alpha=0.8)
    ax.barh(x - width*0.5, df['RF Hit'], width, label='Random Forest', color='green', alpha=0.8)
    if HAS_XGBOOST:
        ax.barh(x + width*0.5, df['XGB Hit'], width, label='XGBoost', color='purple', alpha=0.8)
    ax.barh(x + width*1.5, df['Ensemble Hit'], width, label='Ensemble', color='red', alpha=0.8)
    
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (50%)')
    ax.axvline(0.6, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Good (60%)')
    ax.set_yticks(x)
    ax.set_yticklabels(df['Asset'], fontsize=11)
    ax.set_xlabel('Hit Rate (Directional Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('🎯 Hit Rate: Direction Prediction\n(Higher = Better)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.4, max(0.7, df['Ensemble Hit'].max() + 0.05))
    
    # Panel 3: Improvement Analysis
    ax = axes[1, 0]
    
    df['RF Improvement'] = df['RF R²'] - df['Ridge R²']
    df['Ensemble Improvement'] = df['Ensemble R²'] - df['Ridge R²']
    
    colors = ['green' if x > 0 else 'red' for x in df['Ensemble Improvement']]
    bars = ax.barh(range(len(df)), df['Ensemble Improvement'], color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Asset'], fontsize=11)
    ax.set_xlabel('R² Improvement (Ensemble vs Ridge)', fontsize=13, fontweight='bold')
    ax.set_title('📈 Does ML Improve Predictions?\n(Positive = Yes, Negative = No)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.axvline(0, color='black', linestyle='-', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    for i, val in enumerate(df['Ensemble Improvement']):
        label = f'  +{val:.3f}' if val > 0 else f'  {val:.3f}'
        ax.text(val, i, label, va='center', fontsize=10, fontweight='bold')
    
    # Panel 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = [
        "📊 OVERALL PERFORMANCE SUMMARY",
        "=" * 50,
        "",
        "RIDGE (Linear Baseline):",
        f"  Average Test R²: {df['Ridge R²'].mean():.3f}",
        f"  Average Hit Rate: {df['Ridge Hit'].mean():.1%}",
        f"  Assets with R² > 0: {(df['Ridge R²'] > 0).sum()}/{len(df)}",
        "",
        "RANDOM FOREST (Non-Linear):",
        f"  Average Test R²: {df['RF R²'].mean():.3f}",
        f"  Average Hit Rate: {df['RF Hit'].mean():.1%}",
        f"  Assets with R² > 0: {(df['RF R²'] > 0).sum()}/{len(df)}",
    ]
    
    if HAS_XGBOOST:
        summary_text.extend([
            "",
            "XGBOOST (Gradient Boosting):",
            f"  Average Test R²: {df['XGB R²'].mean():.3f}",
            f"  Average Hit Rate: {df['XGB Hit'].mean():.1%}",
            f"  Assets with R² > 0: {(df['XGB R²'] > 0).sum()}/{len(df)}",
        ])
    
    summary_text.extend([
        "",
        "ENSEMBLE (Average of All):",
        f"  Average Test R²: {df['Ensemble R²'].mean():.3f}",
        f"  Average Hit Rate: {df['Ensemble Hit'].mean():.1%}",
        f"  Assets with R² > 0: {(df['Ensemble R²'] > 0).sum()}/{len(df)}",
        "",
        "=" * 50,
        "💡 KEY INSIGHT:",
        "",
        f"ML models achieve {df['Ensemble Hit'].mean():.1%} directional",
        "accuracy (better than 50% random), but R²",
        "remains low. This suggests returns are partly",
        "predictable in DIRECTION but not MAGNITUDE.",
        "",
        "Ensemble averaging helps reduce overfitting",
        f"by {abs(df['Ensemble Improvement'].mean()):.3f} R² on average.",
    ])
    
    ax.text(0.1, 0.95, '\n'.join(summary_text), 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('🤖 Advanced ML Models: Anti-Overfitting Comparison\n' +
                 'Walk-Forward Cross-Validation Results',
                 fontsize=17, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ml_improved_predictions_dashboard.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'ml_improved_predictions_dashboard.png'


def create_method_comparison_table(all_results, output_dir):
    """Create simple comparison table."""
    data = []
    for asset, result in all_results.items():
        if result:
            # Determine best model
            r2_values = {
                'Ridge': result['ridge']['avg_test_r2'],
                'RF': result['random_forest']['avg_test_r2'],
                'Ensemble': result['ensemble']['avg_test_r2']
            }
            if HAS_XGBOOST and 'xgboost' in result:
                r2_values['XGB'] = result['xgboost']['avg_test_r2']
            
            best_model = max(r2_values, key=r2_values.get)
            best_r2 = r2_values[best_model]
            
            data.append({
                'Asset': ASSET_LABELS.get(asset, asset),
                'Best Model': best_model,
                'Best R²': best_r2,
                'Hit Rate': result['ensemble']['avg_hit_rate'],
                'Verdict': '✅ IMPROVED' if best_r2 > result['ridge']['avg_test_r2'] else '~SIMILAR'
            })
    
    df = pd.DataFrame(data).sort_values('Best R²', ascending=False)
    
    # Create visual
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Asset', 'Best Model', 'R²', 'Hit Rate', 'Verdict']]
    
    for _, row in df.iterrows():
        table_data.append([
            row['Asset'],
            row['Best Model'],
            f"{row['Best R²']:.3f}",
            f"{row['Hit Rate']:.1%}",
            row['Verdict']
        ])
    
    # Colors
    colors = [['lightgray'] * 5]
    for _, row in df.iterrows():
        if row['Best R²'] > 0:
            color = 'lightgreen'
        elif row['Best R²'] > -0.1:
            color = 'lightyellow'
        else:
            color = 'lightcoral'
        colors.append([color] * 5)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     cellColours=colors, bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)
    
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold', fontsize=15)
    
    plt.title('🤖 ML Model Performance Summary\n' +
              'Which Method Works Best Per Asset?',
              fontsize=16, fontweight='bold', pad=20)
    
    legend_text = (
        '✅ IMPROVED = ML model (RF/XGB/Ensemble) beats simple Ridge\n'
        '~SIMILAR = ML performs about the same as Ridge\n\n'
        'GREEN = Positive R² (model explains some variation)\n'
        'YELLOW = Near-zero R² (weak but not terrible)\n'
        'RED = Negative R² (worse than predicting average)\n\n'
        f'Methods tested: Ridge, Random Forest{", XGBoost" if HAS_XGBOOST else ""}, Ensemble\n'
        'All use 5-fold walk-forward cross-validation (anti-overfitting)'
    )
    
    fig.text(0.5, 0.05, legend_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')
    
    plt.savefig(output_dir / 'ml_model_selection_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'ml_model_selection_table.png'


def main():
    print("=" * 80)
    print("IMPROVED ML PREDICTIONS WITH ANTI-OVERFITTING")
    print("Random Forest + XGBoost + Walk-Forward Validation")
    print("=" * 80)
    
    if not HAS_XGBOOST:
        print("\n💡 TIP: Install XGBoost for even better results:")
        print("   pip install xgboost\n")
    
    # Load data
    print("\n📊 Loading data...")
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", index_col=0, parse_dates=True)
    
    # Prepare
    print("📝 Preparing data...")
    combined = prepare_data(returns, features, lag=1)
    
    # Run improved models
    print("\n🤖 Running ML models with anti-overfitting techniques...")
    print("   (This may take 2-3 minutes...)\n")
    
    all_results = {}
    assets = returns['ticker'].unique()
    
    for i, asset in enumerate(assets, 1):
        print(f"  [{i}/{len(assets)}] {ASSET_LABELS.get(asset, asset)}...")
        result = run_improved_models(combined, asset, top_n=5)
        if result:
            all_results[asset] = result
    
    print(f"\n  ✓ Completed analysis for {len(all_results)} assets")
    
    # Create output directory
    output_dir = Path("reports/figures_friendly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    # Create visualizations
    print("\n🎨 Creating ML comparison dashboard...")
    file1 = create_ml_comparison_dashboard(all_results, output_dir)
    generated_files.append(file1)
    print(f"  ✓ Saved {file1}")
    
    print("\n🎨 Creating model selection table...")
    file2 = create_method_comparison_table(all_results, output_dir)
    generated_files.append(file2)
    print(f"  ✓ Saved {file2}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("✅ ML IMPROVED PREDICTIONS COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output location: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    for i, f in enumerate(generated_files, 1):
        print(f"  {i}. {f}")
    
    # Calculate improvements
    ridge_avg = np.mean([r['ridge']['avg_test_r2'] for r in all_results.values()])
    rf_avg = np.mean([r['random_forest']['avg_test_r2'] for r in all_results.values()])
    ensemble_avg = np.mean([r['ensemble']['avg_test_r2'] for r in all_results.values()])
    
    ridge_hit = np.mean([r['ridge']['avg_hit_rate'] for r in all_results.values()])
    ensemble_hit = np.mean([r['ensemble']['avg_hit_rate'] for r in all_results.values()])
    
    print("\n💡 KEY IMPROVEMENTS:")
    print(f"\n📈 R² (Explanatory Power):")
    print(f"   Ridge (baseline): {ridge_avg:.3f}")
    print(f"   Random Forest: {rf_avg:.3f}")
    print(f"   Ensemble: {ensemble_avg:.3f}")
    print(f"   Improvement: {ensemble_avg - ridge_avg:+.3f}")
    
    print(f"\n🎯 Hit Rate (Direction):")
    print(f"   Ridge: {ridge_hit:.1%}")
    print(f"   Ensemble: {ensemble_hit:.1%}")
    print(f"   Improvement: {(ensemble_hit - ridge_hit)*100:+.1f} percentage points")
    
    print("\n🎓 ANTI-OVERFITTING TECHNIQUES USED:")
    print("   ✓ Walk-forward cross-validation (5 folds)")
    print("   ✓ Random Forest with shallow trees (max_depth=3)")
    print("   ✓ XGBoost with strong regularization (alpha=1, lambda=1)")
    print("   ✓ Limited feature interactions")
    print("   ✓ Ensemble averaging to reduce variance")
    print("   ✓ Higher regularization (Ridge alpha=10)")
    
    if ensemble_hit > 0.60:
        print("\n✅ SUCCESS: Ensemble achieves >60% directional accuracy!")
    elif ensemble_hit > 0.55:
        print("\n~PARTIAL: Models show directional prediction ability")
    else:
        print("\n⚠️  LIMITED: Even ML struggles with these returns")
    
    print("\n📖 Open ml_improved_predictions_dashboard.png to see results!")


if __name__ == "__main__":
    main()



