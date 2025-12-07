"""
Comprehensive Backtest: All Assets × All Models
Walk-Forward Reality Check

Compare STATIC test results (optimistic) vs WALK-FORWARD results (realistic)

This answers: "What are the REAL hit rates when we can only use past data?"

Run: python backtesting/test_all_assets_all_models.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed, skipping XGB model")

# ============================================================================
# CONFIGURATION
# ============================================================================

LOOKBACK_WINDOW = 36  # months of training data
MIN_TRAIN_MONTHS = 36  # minimum data before making first prediction

# Define models to test
MODELS_CONFIG = {
    'Ridge': {
        'class': Ridge,
        'params': {'alpha': 10.0, 'random_state': 42}
    },
    'Lasso': {
        'class': Lasso,
        'params': {'alpha': 0.1, 'max_iter': 10000, 'random_state': 42}
    },
    'ElasticNet': {
        'class': ElasticNet,
        'params': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 10000, 'random_state': 42}
    },
    'Random Forest': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 4,
            'min_samples_split': 15,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'random_state': 42
        }
    },
    'Gradient Boosting': {
        'class': GradientBoostingRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'min_samples_split': 15,
            'random_state': 42
        }
    },
    'AdaBoost': {
        'class': AdaBoostRegressor,
        'params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
}

if HAS_XGBOOST:
    MODELS_CONFIG['XGBoost'] = {
        'class': xgb.XGBRegressor,
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'random_state': 42
        }
    }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load returns and features for all assets."""
    returns = pd.read_csv("data_processed/monthly_returns.csv")
    features = pd.read_csv("data_processed/features_monthly.csv", 
                          index_col=0, parse_dates=True)
    
    returns['date'] = pd.to_datetime(returns['date'])
    
    # Get list of all tickers
    tickers = returns['ticker'].unique()
    
    return returns, features, tickers

def prepare_asset_data(returns_df, features_df, ticker):
    """Prepare data for a specific asset."""
    # Filter for ticker
    asset_returns = returns_df[returns_df['ticker'] == ticker].copy()
    asset_returns = asset_returns.sort_values('date').set_index('date')
    
    # Merge with lagged features (use previous month's data to predict this month)
    features_lagged = features_df.shift(1)
    combined = asset_returns.join(features_lagged, how='inner')
    combined = combined.dropna()
    
    return combined

# ============================================================================
# WALK-FORWARD BACKTESTING
# ============================================================================

def walk_forward_backtest(combined, feature_cols, model_config):
    """
    Run walk-forward backtest with rolling window.
    
    Returns:
        predictions, actuals, hit_rate
    """
    predictions = []
    actuals = []
    
    for i in range(MIN_TRAIN_MONTHS, len(combined)):
        # Define training window (rolling 36 months)
        train_start = i - LOOKBACK_WINDOW
        train_end = i
        
        # Get training data
        train_data = combined.iloc[train_start:train_end]
        X_train = train_data[feature_cols].values
        y_train = train_data['return'].values
        
        # Get current month data (to predict)
        current_data = combined.iloc[i:i+1]
        X_current = current_data[feature_cols].values
        y_actual = current_data['return'].values[0]
        
        # Train model
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_current_scaled = scaler.transform(X_current)
            
            # Create and train model
            model = model_config['class'](**model_config['params'])
            model.fit(X_train_scaled, y_train)
            
            # Make prediction
            y_pred = model.predict(X_current_scaled)[0]
            
            predictions.append(y_pred)
            actuals.append(y_actual)
            
        except Exception as e:
            # If model fails, skip this prediction
            continue
    
    if len(predictions) == 0:
        return None, None, 0.0, 0
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate hit rate
    hits = np.sign(predictions) == np.sign(actuals)
    hit_rate = hits.mean() * 100
    
    return predictions, actuals, hit_rate, len(predictions)

# ============================================================================
# STATIC TEST (for comparison)
# ============================================================================

def static_backtest(combined, feature_cols, model_config):
    """
    Traditional 80/20 train-test split (optimistic).
    
    This is what most of the MODEL_GUIDE results are based on.
    """
    if len(combined) < MIN_TRAIN_MONTHS:
        return 0.0, 0
    
    # 80/20 split
    split_idx = int(len(combined) * 0.8)
    train_data = combined.iloc[:split_idx]
    test_data = combined.iloc[split_idx:]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['return'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['return'].values
    
    try:
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = model_config['class'](**model_config['params'])
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate hit rate
        hits = np.sign(y_pred) == np.sign(y_test)
        hit_rate = hits.mean() * 100
        
        return hit_rate, len(y_test)
        
    except Exception as e:
        return 0.0, 0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_backtest():
    """Run backtest for all assets and all models."""
    print("=" * 80)
    print("COMPREHENSIVE BACKTEST: STATIC vs WALK-FORWARD")
    print("=" * 80)
    print("\nTesting all assets with all models...")
    print("This will take a few minutes...\n")
    
    # Load data
    returns, features, tickers = load_all_data()
    feature_cols = features.columns.tolist()
    
    print(f"Assets to test: {len(tickers)}")
    print(f"Models to test: {len(MODELS_CONFIG)}")
    print(f"Total combinations: {len(tickers) * len(MODELS_CONFIG)}\n")
    
    # Results storage
    results = []
    
    # Test each asset
    for ticker_idx, ticker in enumerate(tickers, 1):
        print(f"\n[{ticker_idx}/{len(tickers)}] Testing {ticker}...")
        
        # Prepare data
        try:
            combined = prepare_asset_data(returns, features, ticker)
        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            continue
        
        if len(combined) < MIN_TRAIN_MONTHS * 2:
            print(f"  ⚠️  Insufficient data ({len(combined)} months)")
            continue
        
        print(f"  Data: {combined.index.min().strftime('%Y-%m')} to {combined.index.max().strftime('%Y-%m')} ({len(combined)} months)")
        
        # Test each model
        for model_name, model_config in MODELS_CONFIG.items():
            try:
                # Static test (optimistic)
                static_hit_rate, static_n = static_backtest(combined, feature_cols, model_config)
                
                # Walk-forward test (realistic)
                _, _, walkforward_hit_rate, walkforward_n = walk_forward_backtest(
                    combined, feature_cols, model_config
                )
                
                # Calculate degradation
                degradation = static_hit_rate - walkforward_hit_rate
                degradation_pct = (degradation / static_hit_rate * 100) if static_hit_rate > 0 else 0
                
                results.append({
                    'ticker': ticker,
                    'model': model_name,
                    'static_hit_rate': static_hit_rate,
                    'static_n': static_n,
                    'walkforward_hit_rate': walkforward_hit_rate,
                    'walkforward_n': walkforward_n,
                    'degradation': degradation,
                    'degradation_pct': degradation_pct,
                    'data_months': len(combined)
                })
                
                # Print result
                symbol = "✅" if walkforward_hit_rate > 55 else "⚠️" if walkforward_hit_rate > 50 else "❌"
                print(f"    {symbol} {model_name:18s}: Static={static_hit_rate:5.1f}%  →  Walk-Forward={walkforward_hit_rate:5.1f}%  (Δ {degradation:+.1f}%)")
                
            except Exception as e:
                print(f"    ❌ {model_name}: Error - {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def create_reality_check_report(results_df):
    """Create comprehensive visualization comparing static vs walk-forward."""
    
    # Calculate summary statistics
    summary = results_df.groupby('model').agg({
        'static_hit_rate': 'mean',
        'walkforward_hit_rate': 'mean',
        'degradation': 'mean',
        'degradation_pct': 'mean'
    }).round(1)
    
    print("\n" + "=" * 80)
    print("SUMMARY: AVERAGE HIT RATES BY MODEL")
    print("=" * 80)
    print(summary.to_string())
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # ========================================================================
    # Panel 1: Hit Rate Comparison - All Assets
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Pivot for heatmap
    pivot_static = results_df.pivot(index='ticker', columns='model', values='static_hit_rate')
    pivot_walkforward = results_df.pivot(index='ticker', columns='model', values='walkforward_hit_rate')
    
    x = np.arange(len(results_df['ticker'].unique()))
    width = 0.35
    
    # Group by ticker and calculate average across all models
    avg_by_ticker = results_df.groupby('ticker')[['static_hit_rate', 'walkforward_hit_rate']].mean()
    avg_by_ticker = avg_by_ticker.sort_values('walkforward_hit_rate', ascending=False)
    
    x_pos = np.arange(len(avg_by_ticker))
    ax1.bar(x_pos - width/2, avg_by_ticker['static_hit_rate'], width, 
           label='Static Test (Optimistic)', color='#F4A261', alpha=0.8)
    ax1.bar(x_pos + width/2, avg_by_ticker['walkforward_hit_rate'], width,
           label='Walk-Forward (Realistic)', color='#2A9D8F', alpha=0.8)
    
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random (50%)', alpha=0.5)
    ax1.axhline(y=60, color='green', linestyle='--', linewidth=2, label='Strong (60%)', alpha=0.5)
    
    ax1.set_xlabel('Asset', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Hit Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Hit Rate Comparison: Static vs Walk-Forward Testing\n(Averaged Across All Models)',
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(avg_by_ticker.index, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)
    
    # ========================================================================
    # Panel 2: Degradation Analysis
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    avg_degradation = results_df.groupby('ticker')['degradation'].mean().sort_values()
    colors = ['red' if x > 0 else 'green' for x in avg_degradation.values]
    
    ax2.barh(range(len(avg_degradation)), avg_degradation.values, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(avg_degradation)))
    ax2.set_yticklabels(avg_degradation.index)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Hit Rate Degradation (Static - Walk-Forward)', fontsize=12, fontweight='bold')
    ax2.set_title('Reality Check: How Much Did Hit Rate Drop?\n(Negative = Walk-Forward Better)',
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ========================================================================
    # Panel 3: Model Performance Comparison
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    model_avg = results_df.groupby('model')[['static_hit_rate', 'walkforward_hit_rate']].mean()
    model_avg = model_avg.sort_values('walkforward_hit_rate', ascending=False)
    
    x_models = np.arange(len(model_avg))
    ax3.bar(x_models - width/2, model_avg['static_hit_rate'], width,
           label='Static', color='#F4A261', alpha=0.8)
    ax3.bar(x_models + width/2, model_avg['walkforward_hit_rate'], width,
           label='Walk-Forward', color='#2A9D8F', alpha=0.8)
    
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Hit Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Which Models Survive Reality?\n(Average Across All Assets)',
                 fontsize=13, fontweight='bold')
    ax3.set_xticks(x_models)
    ax3.set_xticklabels(model_avg.index, rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(40, 70)
    
    # ========================================================================
    # Panel 4: Scatter - Static vs Walk-Forward
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.scatter(results_df['static_hit_rate'], results_df['walkforward_hit_rate'],
               alpha=0.5, s=50)
    
    # Add diagonal line (perfect agreement)
    ax4.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Agreement')
    
    # Add 50% lines
    ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Static Hit Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Walk-Forward Hit Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Static vs Walk-Forward: Asset × Model Combinations\n(Points Below Line = Overfitting)',
                 fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(30, 100)
    ax4.set_ylim(30, 100)
    
    # Add quadrant labels
    ax4.text(70, 40, 'OVERFITTING\n(Static good, Walk bad)', 
            ha='center', va='center', fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.text(70, 70, 'GENUINE SKILL\n(Both good)',
            ha='center', va='center', fontsize=10, color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # Panel 5: Best Performers (Walk-Forward)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Get top 15 by walk-forward hit rate
    top_performers = results_df.nlargest(15, 'walkforward_hit_rate')
    top_performers['label'] = top_performers['ticker'] + '\n' + top_performers['model']
    
    colors_perf = ['#06A77D' if x > 60 else '#F4D35E' if x > 55 else '#EE6C4D' 
                  for x in top_performers['walkforward_hit_rate']]
    
    y_pos = range(len(top_performers))
    ax5.barh(y_pos, top_performers['walkforward_hit_rate'], color=colors_perf, alpha=0.8)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(top_performers['label'], fontsize=9)
    ax5.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax5.axvline(x=60, color='green', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_xlabel('Walk-Forward Hit Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('🏆 Top 15 Performers (Realistic Testing)',
                 fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.invert_yaxis()
    
    # ========================================================================
    # Panel 6: Summary Table
    # ========================================================================
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate key statistics
    total_tests = len(results_df)
    avg_static = results_df['static_hit_rate'].mean()
    avg_walkforward = results_df['walkforward_hit_rate'].mean()
    avg_degradation = results_df['degradation'].mean()
    
    # Count successes
    static_above_60 = (results_df['static_hit_rate'] > 60).sum()
    walkforward_above_60 = (results_df['walkforward_hit_rate'] > 60).sum()
    walkforward_above_55 = (results_df['walkforward_hit_rate'] > 55).sum()
    walkforward_above_50 = (results_df['walkforward_hit_rate'] > 50).sum()
    
    # Best performers
    best_static = results_df.nlargest(1, 'static_hit_rate').iloc[0]
    best_walkforward = results_df.nlargest(1, 'walkforward_hit_rate').iloc[0]
    
    table_data = [
        ['Metric', 'Static Test', 'Walk-Forward', 'Reality Check'],
        ['Average Hit Rate', f"{avg_static:.1f}%", f"{avg_walkforward:.1f}%", 
         f"{avg_degradation:+.1f}% drop"],
        ['Above 60% (Strong)', f"{static_above_60} ({static_above_60/total_tests*100:.0f}%)",
         f"{walkforward_above_60} ({walkforward_above_60/total_tests*100:.0f}%)",
         f"{static_above_60 - walkforward_above_60} fewer"],
        ['Above 55% (Good)', '-',
         f"{walkforward_above_55} ({walkforward_above_55/total_tests*100:.0f}%)", '-'],
        ['Above 50% (Better than Random)', '-',
         f"{walkforward_above_50} ({walkforward_above_50/total_tests*100:.0f}%)", '-'],
        ['Best Performer', 
         f"{best_static['ticker']} + {best_static['model']}\n{best_static['static_hit_rate']:.1f}%",
         f"{best_walkforward['ticker']} + {best_walkforward['model']}\n{best_walkforward['walkforward_hit_rate']:.1f}%",
         '-'],
        ['Total Tests', str(total_tests), str(total_tests), '-']
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#264653')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    # Overall title
    fig.suptitle('🔍 REALITY CHECK: Static Testing vs Walk-Forward Validation\n' + 
                 'How Much Did Hit Rates Drop When We Removed Future Information?',
                fontsize=18, fontweight='bold', y=0.995)
    
    return fig, results_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Run backtest
    results_df = run_comprehensive_backtest()
    
    if len(results_df) == 0:
        print("❌ No results generated!")
        return
    
    # Create visualization
    print("\n📊 Creating comprehensive report...")
    fig, results_df = create_reality_check_report(results_df)
    
    # Save results
    output_dir = Path("backtesting/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    fig_path = output_dir / 'reality_check_all_assets_models.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved visualization: {fig_path}")
    
    # Save detailed CSV
    csv_path = output_dir / 'reality_check_detailed_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✅ Saved detailed results: {csv_path}")
    
    # Save summary by model
    model_summary = results_df.groupby('model').agg({
        'static_hit_rate': ['mean', 'std', 'min', 'max'],
        'walkforward_hit_rate': ['mean', 'std', 'min', 'max'],
        'degradation': ['mean', 'std']
    }).round(1)
    
    summary_path = output_dir / 'reality_check_model_summary.csv'
    model_summary.to_csv(summary_path)
    print(f"✅ Saved model summary: {summary_path}")
    
    # Save summary by asset
    asset_summary = results_df.groupby('ticker').agg({
        'static_hit_rate': ['mean', 'std', 'min', 'max'],
        'walkforward_hit_rate': ['mean', 'std', 'min', 'max'],
        'degradation': ['mean', 'std']
    }).round(1)
    
    asset_summary_path = output_dir / 'reality_check_asset_summary.csv'
    asset_summary.to_csv(asset_summary_path)
    print(f"✅ Saved asset summary: {asset_summary_path}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    avg_static = results_df['static_hit_rate'].mean()
    avg_walkforward = results_df['walkforward_hit_rate'].mean()
    avg_degradation = results_df['degradation'].mean()
    
    print(f"\n📊 Overall Results:")
    print(f"  • Average Static Hit Rate: {avg_static:.1f}%")
    print(f"  • Average Walk-Forward Hit Rate: {avg_walkforward:.1f}%")
    print(f"  • Average Degradation: {avg_degradation:.1f} percentage points")
    print(f"  • Relative Drop: {avg_degradation/avg_static*100:.1f}%")
    
    walkforward_above_55 = (results_df['walkforward_hit_rate'] > 55).sum()
    walkforward_above_60 = (results_df['walkforward_hit_rate'] > 60).sum()
    total = len(results_df)
    
    print(f"\n🎯 Walk-Forward Performance:")
    print(f"  • Above 60% (Strong): {walkforward_above_60}/{total} ({walkforward_above_60/total*100:.1f}%)")
    print(f"  • Above 55% (Good): {walkforward_above_55}/{total} ({walkforward_above_55/total*100:.1f}%)")
    
    # Best performers
    top_5 = results_df.nlargest(5, 'walkforward_hit_rate')
    print(f"\n🏆 Top 5 Real-World Performers:")
    for idx, row in top_5.iterrows():
        print(f"  {row['ticker']:6s} + {row['model']:18s}: {row['walkforward_hit_rate']:.1f}% (was {row['static_hit_rate']:.1f}%)")
    
    print("\n✅ Reality check complete! See visualization for full analysis.\n")

if __name__ == "__main__":
    main()



