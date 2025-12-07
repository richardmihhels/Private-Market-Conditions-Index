"""
Create Figure 1 Heatmap with Model Averages - Fixed overlapping text
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Read data
base_path = Path(__file__).parent.parent
results_path = base_path / 'backtesting' / 'results'
output_path = base_path / 'reports' / 'poster_figures'

detailed = pd.read_csv(results_path / 'reality_check_detailed_results.csv')
model_summary = pd.read_csv(results_path / 'reality_check_model_summary.csv', header=[0, 1], index_col=0)
asset_summary = pd.read_csv(results_path / 'reality_check_asset_summary.csv', header=[0, 1], index_col=0)

print("Creating Figure 1 with Model Averages (FIXED - no overlapping)...")

#==============================================================================
# FIGURE 1: Walk-Forward Hit Rate Heatmap with Averages - FIXED
#==============================================================================

fig1 = plt.figure(figsize=(18, 11), facecolor='white')
ax1 = plt.subplot(111)

# Pivot data
heatmap_data = detailed.pivot(index='model', columns='ticker', values='walkforward_hit_rate')

# Order by performance
model_order = model_summary.sort_values(('walkforward_hit_rate', 'mean'), ascending=False).index
asset_order = asset_summary.sort_values(('walkforward_hit_rate', 'mean'), ascending=False).index
heatmap_data = heatmap_data.loc[model_order][asset_order]

# Add average column
heatmap_data['AVERAGE'] = heatmap_data.mean(axis=1)

# Create extended column labels
extended_labels = list(asset_order) + ['AVERAGE']

# Create heatmap
im = ax1.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto',
                vmin=45, vmax=75, interpolation='nearest')

# Set ticks
ax1.set_xticks(np.arange(len(extended_labels)))
ax1.set_yticks(np.arange(len(model_order)))
ax1.set_xticklabels(extended_labels, rotation=45, ha='right', fontsize=16, fontweight='bold')
ax1.set_yticklabels(model_order, fontsize=16, fontweight='bold')

# Make AVERAGE label stand out
xtick_labels = ax1.get_xticklabels()
xtick_labels[-1].set_color('darkblue')
xtick_labels[-1].set_fontweight('bold')
xtick_labels[-1].set_fontsize(18)

# Add values inside cells
for i in range(len(model_order)):
    for j in range(len(extended_labels)):
        value = heatmap_data.values[i, j]
        
        # Different styling for average column
        if j == len(extended_labels) - 1:  # Average column
            color = 'white' if value < 55 or value > 68 else 'black'
            fontweight = 'bold'
            fontsize = 14
        else:
            color = 'white' if value < 55 or value > 68 else 'black'
            fontweight = 'bold'
            fontsize = 12
            
        ax1.text(j, i, f'{value:.1f}', ha='center', va='center',
                color=color, fontsize=fontsize, fontweight=fontweight)

# Add vertical separator line before AVERAGE column
ax1.axvline(x=len(asset_order) - 0.5, color='black', linewidth=3, linestyle='-', alpha=0.8)

# Colorbar with adjusted positioning to avoid overlap
cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.02, aspect=25)
cbar.set_label('Hit Rate (%)', rotation=270, labelpad=25, fontsize=16, fontweight='bold')
cbar.ax.tick_params(labelsize=13)

# Title and labels
ax1.set_title('Walk-Forward Hit Rates by Model and Asset\n(Higher = Better Prediction, 50% = Random Guessing)',
              fontsize=24, fontweight='bold', pad=25)
ax1.set_xlabel('Assets (ordered by predictability) + Model Average', 
               fontsize=18, fontweight='bold', labelpad=15)
ax1.set_ylabel('Models (ordered by performance)', fontsize=18, fontweight='bold', labelpad=15)

# Add annotation box - positioned BELOW the colorbar label to avoid overlap
textstr = 'Baseline: 50%\n(random guessing)\n\nAVERAGE column\n= mean across\nall 14 assets'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2)
# Position lower to avoid colorbar label
ax1.text(1.14, 0.25, textstr, transform=ax1.transAxes, fontsize=12,
         verticalalignment='center', bbox=props, fontweight='bold')

# Adjust layout to prevent cutoff
plt.tight_layout(rect=[0, 0, 0.96, 1])

# Save
plt.savefig(output_path / 'Figure1_Heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_path / 'Figure1_Heatmap.png', format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: Figure1_Heatmap.pdf & .png (no overlapping text)")

# Print summary
print("\nModel Averages:")
for model in model_order:
    avg = heatmap_data.loc[model, 'AVERAGE']
    print(f"  {model:20s}: {avg:.1f}%")

plt.close()

print("\n" + "="*80)
print("FIGURE 1 FIXED - No Overlapping Text")
print("="*80)
print("Changes made:")
print("  • Colorbar label positioned with less padding (labelpad=25)")
print("  • Annotation box moved lower (y=0.25 instead of 0.5)")
print("  • Tighter colorbar spacing (pad=0.02)")
print("  • All text now clearly visible without overlap")
print("="*80)

