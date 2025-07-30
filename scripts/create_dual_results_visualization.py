#!/usr/bin/env python3
"""
Create a dual visualization for the Filo-Transformer results section.
Two side-by-side plots showing:
1. Performance comparison across metrics
2. Fusion weights visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set the style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configure fonts for LaTeX compatibility
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13
})

# Data from the results
metrics = ['Accuracy', 'AUC', 'Recall', 'F1-Score']
baseline_scores = [0.8671, 0.8882, 0.7605, 0.7790]
filo_scores = [0.8702, 0.9071, 0.7661, 0.7847]

# Calculate improvements
improvements = [(filo - base) * 100 for filo, base in zip(filo_scores, baseline_scores)]

# Fusion weights data
fusion_weights = {
    'Phylogenetic\nFeatures': 65,
    'Semantic\nFeatures': 35
}

# Create figure with two subplots
fig = plt.figure(figsize=(12, 5))

# Colors
baseline_color = '#3498db'  # Blue
filo_color = '#e74c3c'      # Red
phylo_color = '#2ecc71'     # Green
semantic_color = '#f39c12'  # Orange

# ========== Left subplot: Performance Comparison ==========
ax1 = plt.subplot(1, 2, 1)

x = np.arange(len(metrics))
width = 0.35

# Create bars
bars1 = ax1.bar(x - width/2, baseline_scores, width, label='Baseline (GPT + FT)', 
                 color=baseline_color, alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax1.bar(x + width/2, filo_scores, width, label='Filo-Transformer', 
                 color=filo_color, alpha=0.8, edgecolor='black', linewidth=1.2)

# Add value labels on bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1, baseline_scores)
add_value_labels(bars2, filo_scores)

# Add improvement indicators
for i, (x_pos, improvement) in enumerate(zip(x, improvements)):
    # Draw arrow and improvement percentage
    y_base = baseline_scores[i] + 0.02
    y_filo = filo_scores[i] + 0.02
    
    if improvement > 0:
        ax1.annotate('', xy=(x_pos + width/2, y_filo), xytext=(x_pos - width/2, y_base),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, alpha=0.7))
        
        # Add improvement text
        mid_y = (y_base + y_filo) / 2
        ax1.text(x_pos, mid_y + 0.015, f'+{improvement:.2f}%', 
                ha='center', va='center', fontsize=8, color='green', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='green', alpha=0.8))

# Customize the plot
ax1.set_xlabel('Metrics', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('(a) Performance Comparison: Baseline vs Filo-Transformer', 
              fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax1.set_ylim(0.7, 0.95)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add subtle background shading for better readability
ax1.axhspan(0.85, 0.95, alpha=0.05, color='green', label='High Performance Zone')
ax1.axhspan(0.75, 0.85, alpha=0.05, color='yellow')
ax1.axhspan(0.7, 0.75, alpha=0.05, color='red')

# ========== Right subplot: Fusion Weights Visualization ==========
ax2 = plt.subplot(1, 2, 2)

# Create a more sophisticated visualization for fusion weights
# Using a semi-circular gauge chart

# Create pie chart with custom styling
colors = [phylo_color, semantic_color]
wedges, texts, autotexts = ax2.pie(fusion_weights.values(), 
                                    labels=None,  # We'll add custom labels
                                    colors=colors,
                                    autopct='%1.0f%%',
                                    startangle=90,
                                    counterclock=False,
                                    wedgeprops=dict(width=0.5, edgecolor='black', linewidth=2),
                                    textprops={'fontsize': 14, 'fontweight': 'bold'})

# Customize the percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)

# Add custom labels with icons
label_distance = 1.3
for i, (label, value) in enumerate(fusion_weights.items()):
    angle = wedges[i].theta1 + (wedges[i].theta2 - wedges[i].theta1) / 2
    x = label_distance * np.cos(np.radians(angle))
    y = label_distance * np.sin(np.radians(angle))
    
    # Add label with background
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor=colors[i], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.text(x, y, label, ha='center', va='center', fontweight='bold',
            fontsize=11, bbox=bbox_props, color='white')

# Add center text
ax2.text(0, 0, 'Learned\nFusion\nWeights', ha='center', va='center', 
         fontsize=12, fontweight='bold', 
         bbox=dict(boxstyle='circle,pad=0.5', facecolor='white', 
                  edgecolor='black', linewidth=2))

# Add title
ax2.set_title('(b) Automatic Feature Importance Learning', 
              fontweight='bold', pad=15)

# Add annotation explaining the significance
ax2.text(0, -1.5, 'Model automatically learned to prioritize phylogenetic features (65%)\n' + 
         'over semantic features (35%), validating the importance of\n' + 
         'cascade structure in rumor detection',
         ha='center', va='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                  edgecolor='gray', alpha=0.8))

# Make the plot square
ax2.set_aspect('equal')

# Adjust layout
plt.tight_layout()

# Add overall figure title
fig.suptitle('Filo-Transformer: Comprehensive Performance Analysis', 
             fontsize=14, fontweight='bold', y=1.02)

# Save the figure
output_path = '/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_visualization.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Dual visualization saved to: {output_path}")

# Also save as PNG for preview
output_path_png = output_path.replace('.pdf', '.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', format='png')
print(f"PNG version saved to: {output_path_png}")

# Create a second version with more detailed annotations
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# ========== Alternative Left subplot: Grouped Performance ==========
# Group metrics by type
accuracy_metrics = ['Accuracy', 'F1-Score']
discrimination_metrics = ['AUC', 'Recall']

# Prepare data for grouped visualization
baseline_acc = [0.8671, 0.7790]
filo_acc = [0.8702, 0.7847]
baseline_disc = [0.8882, 0.7605]
filo_disc = [0.9071, 0.7661]

# Create grouped bars
x_acc = np.arange(len(accuracy_metrics))
x_disc = np.arange(len(discrimination_metrics)) + len(accuracy_metrics) + 0.5

# Plot accuracy metrics
bars_acc1 = ax3.bar(x_acc - width/2, baseline_acc, width, label='Baseline', 
                    color=baseline_color, alpha=0.8, edgecolor='black')
bars_acc2 = ax3.bar(x_acc + width/2, filo_acc, width, label='Filo-Transformer', 
                    color=filo_color, alpha=0.8, edgecolor='black')

# Plot discrimination metrics
bars_disc1 = ax3.bar(x_disc - width/2, baseline_disc, width, 
                     color=baseline_color, alpha=0.8, edgecolor='black')
bars_disc2 = ax3.bar(x_disc + width/2, filo_disc, width, 
                     color=filo_color, alpha=0.8, edgecolor='black')

# Add group labels
ax3.text(np.mean(x_acc), 0.68, 'Overall Performance', ha='center', va='center',
         fontsize=10, fontweight='bold', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
ax3.text(np.mean(x_disc), 0.68, 'Discrimination Ability', ha='center', va='center',
         fontsize=10, fontweight='bold', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

# Set labels
all_labels = accuracy_metrics + discrimination_metrics
ax3.set_xticks(np.concatenate([x_acc, x_disc]))
ax3.set_xticklabels(all_labels, rotation=15, ha='right')
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('(a) Performance Analysis by Metric Category', fontweight='bold', pad=15)
ax3.legend(loc='lower right', frameon=True, fancybox=True)
ax3.set_ylim(0.65, 0.95)
ax3.grid(True, alpha=0.3, axis='y')

# Add vertical separator
ax3.axvline(x=len(accuracy_metrics) + 0.25, color='gray', linestyle='--', alpha=0.5)

# ========== Alternative Right subplot: Radar Chart ==========
# Create radar chart for comprehensive comparison
categories = ['Accuracy', 'AUC', 'Recall', 'F1-Score']
N = len(categories)

# Normalize scores to 0-1 range for better visualization
min_score = 0.75
max_score = 0.92
baseline_norm = [(score - min_score) / (max_score - min_score) for score in baseline_scores]
filo_norm = [(score - min_score) / (max_score - min_score) for score in filo_scores]

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Close the plot
baseline_norm += baseline_norm[:1]
filo_norm += filo_norm[:1]

# Initialize the spider plot
ax4 = plt.subplot(1, 2, 2, projection='polar')

# Draw one axis per variable and add labels
plt.xticks(angles[:-1], categories, size=10)

# Draw ylabels
ax4.set_rlabel_position(0)
yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
yticklabels = [f'{min_score + y * (max_score - min_score):.2f}' for y in yticks]
plt.yticks(yticks, yticklabels, color="grey", size=8)
plt.ylim(0, 1)

# Plot data
ax4.plot(angles, baseline_norm, 'o-', linewidth=2, label='Baseline', color=baseline_color)
ax4.fill(angles, baseline_norm, alpha=0.25, color=baseline_color)

ax4.plot(angles, filo_norm, 'o-', linewidth=2, label='Filo-Transformer', color=filo_color)
ax4.fill(angles, filo_norm, alpha=0.25, color=filo_color)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.set_title('(b) Multi-Metric Performance Radar', fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save alternative version
output_path2 = '/home/acauan/ufam/papers/01_sbseg_filo_trans/visualizations/dual_results_alternative.pdf'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_path2.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
print(f"Alternative visualization saved to: {output_path2}")

plt.show()