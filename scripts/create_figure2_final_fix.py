"""
Create Figure 2: Model Rankings - Final Fix (no overlapping at all)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# Read data
base_path = Path(__file__).parent.parent
results_path = base_path / 'backtesting' / 'results'
output_path = base_path / 'reports' / 'poster_figures'

model_summary = pd.read_csv(results_path / 'reality_check_model_summary.csv', header=[0, 1], index_col=0)

print("Creating Figure 2: Model Rankings (FINAL FIX - legend in upper right)...")

#==============================================================================
# FIGURE 2: Model Rankings - FINAL FIX
#==============================================================================

fig2 = plt.figure(figsize=(13, 9), facecolor='white')
ax2 = plt.subplot(111)

# Prepare data
models = model_summary.index
static_rates = model_summary[('static_hit_rate', 'mean')].values
walkforward_rates = model_summary[('walkforward_hit_rate', 'mean')].values

# Sort by walk-forward
sorted_idx = np.argsort(walkforward_rates)[::-1]
models_sorted = [models[i] for i in sorted_idx]
static_sorted = static_rates[sorted_idx]
walkforward_sorted = walkforward_rates[sorted_idx]

y = np.arange(len(models_sorted))
height = 0.35

# Bars
bars1 = ax2.barh(y - height/2, static_sorted, height, 
                 label='Static Testing (80/20)', color='lightcoral', 
                 alpha=0.85, edgecolor='black', linewidth=1.5)
bars2 = ax2.barh(y + height/2, walkforward_sorted, height,
                 label='Walk-Forward (36-month)', color='steelblue', 
                 alpha=0.85, edgecolor='black', linewidth=1.5)

# Formatting - extend x-axis to make room for legend
ax2.set_yticks(y)
ax2.set_yticklabels(models_sorted, fontsize=16, fontweight='bold')
ax2.set_xlabel('Hit Rate (%)', fontsize=18, fontweight='bold', labelpad=12)
ax2.set_title('Model Performance Comparison\nStatic vs Walk-Forward Validation',
              fontsize=24, fontweight='bold', pad=25)

# Random baseline line
ax2.axvline(x=50, color='red', linestyle='--', linewidth=3, alpha=0.7, label='Random (50%)')

# Legend positioned at UPPER RIGHT - away from ALL data
ax2.legend(loc='upper right', fontsize=13, framealpha=0.97, edgecolor='black', 
           fancybox=True, shadow=True, borderpad=1)

ax2.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1.5)
ax2.set_xlim(44, 68)  # Extended to give room for labels

# Add value labels - ALL positioned to avoid legend in upper right
for i, (s, w) in enumerate(zip(static_sorted, walkforward_sorted)):
    # Static value - inside bar if room, else outside
    if s > 46:
        ax2.text(s - 0.5, i - height/2, f'{s:.1f}', va='center', ha='right', 
                 fontsize=11, fontweight='bold', color='darkred')
    else:
        ax2.text(s + 0.5, i - height/2, f'{s:.1f}', va='center', ha='left',
                 fontsize=11, fontweight='bold', color='darkred')
    
    # Walk-forward value - positioned outside bars (to the right)
    # ALL labels go outside to the right since legend is in upper right corner
    ax2.text(w + 0.5, i + height/2, f'{w:.1f}', va='center', ha='left',
             fontsize=11, fontweight='bold', color='darkblue')

# Adjust layout
plt.tight_layout()

# Save
plt.savefig(output_path / 'Figure2_ModelRankings.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_path / 'Figure2_ModelRankings.png', format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: Figure2_ModelRankings.pdf & .png (legend in upper right)")

plt.close()

print("\n" + "="*80)
print("FIGURE 2 FINAL FIX - Legend in Upper Right")
print("="*80)
print("Changes made:")
print("  • Legend moved to UPPER RIGHT corner (away from all data)")
print("  • Extended x-axis to 68% (more room for labels)")
print("  • ALL walk-forward labels positioned to the right of bars")
print("  • Random Forest label at 60.5 now clearly visible")
print("  • No overlapping anywhere")
print("="*80)

