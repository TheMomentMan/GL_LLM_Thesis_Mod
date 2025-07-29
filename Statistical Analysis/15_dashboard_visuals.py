#!/usr/bin/env python3
"""
Professional Tukey HSD Visualizations for Trust Dimensions
==========================================================
Creates publication-quality visualizations for thesis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def create_tukey_visualizations():
    """Create comprehensive Tukey HSD visualizations."""
    
    # Data from your results
    trust_dimensions = ['Source citations', 'Confidence score', 'Clarity', 
                       'technical language', 'Friendly tones', 'Hedging language']
    
    means = [3.904, 3.888, 3.864, 3.704, 3.352, 3.304]
    stds = [1.058, 0.935, 1.042, 1.070, 1.233, 1.123]
    groups = ['A', 'A', 'A', 'A', 'B', 'B']
    n = 125
    
    # Calculate standard errors
    ses = [std/np.sqrt(n) for std in stds]
    
    # Create the comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # Set up a professional color scheme
    colors = {'A': '#2E86AB', 'B': '#A23B72'}  # Blue for high, Pink for low
    
    # 1. Main Plot: Means with Error Bars and Group Letters (Top Left)
    ax1 = plt.subplot(3, 3, (1, 2))
    
    # Create bars with group colors
    bar_colors = [colors[group] for group in groups]
    bars = ax1.bar(range(len(trust_dimensions)), means, 
                   yerr=ses, capsize=8, alpha=0.8, 
                   color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # Add group letters above bars
    for i, (bar, group, mean, se) in enumerate(zip(bars, groups, means, ses)):
        ax1.text(bar.get_x() + bar.get_width()/2, mean + se + 0.08,
                group, ha='center', va='bottom', fontweight='bold', 
                fontsize=16, color='black')
        
        # Add mean values on bars
        ax1.text(bar.get_x() + bar.get_width()/2, mean/2,
                f'{mean:.2f}', ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
    
    # Formatting
    ax1.set_xticks(range(len(trust_dimensions)))
    ax1.set_xticklabels([dim.replace(' ', '\n') for dim in trust_dimensions], 
                        fontsize=12, ha='center')
    ax1.set_ylabel('Mean Trust Rating', fontsize=14, fontweight='bold')
    ax1.set_title('Trust Dimensions: Tukey HSD Groups', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylim(0, 4.5)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at critical difference
    critical_line = max(means) - 0.305  # Critical value from your results
    ax1.axhline(y=critical_line, color='red', linestyle=':', alpha=0.7, 
                label=f'Critical Difference = 0.305')
    ax1.legend(loc='upper right')
    
    # Add group legend
    legend_elements = [mpatches.Patch(color=colors['A'], label='Group A (High Trust)'),
                      mpatches.Patch(color=colors['B'], label='Group B (Lower Trust)')]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    # 2. Significance Matrix Heatmap (Top Right)
    ax2 = plt.subplot(3, 3, 3)
    
    # Create significance matrix from your results
    sig_pairs = [
        ('Clarity', 'Friendly tones'), ('Clarity', 'Hedging language'),
        ('Confidence score', 'Friendly tones'), ('Confidence score', 'Hedging language'),
        ('Source citations', 'Friendly tones'), ('Source citations', 'Hedging language'),
        ('Friendly tones', 'technical language'), ('technical language', 'Hedging language')
    ]
    
    # Create matrix
    n_dims = len(trust_dimensions)
    sig_matrix = np.zeros((n_dims, n_dims))
    
    for pair in sig_pairs:
        i = trust_dimensions.index(pair[0])
        j = trust_dimensions.index(pair[1])
        sig_matrix[i, j] = 1
        sig_matrix[j, i] = 1
    
    # Plot heatmap
    sns.heatmap(sig_matrix, annot=True, fmt='.0f', cmap='RdYlBu_r',
                xticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                yticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                ax=ax2, cbar_kws={'label': 'Significant Difference'})
    ax2.set_title('Pairwise Significance Matrix\n(1 = Significant, 0 = Not Significant)', 
                  fontsize=14, fontweight='bold')
    
    # 3. Forest Plot with Confidence Intervals (Middle Left)
    ax3 = plt.subplot(3, 3, 4)
    
    # Calculate 95% CIs
    from scipy.stats import t
    t_crit = t.ppf(0.975, n-1)
    ci_lower = [mean - t_crit * se for mean, se in zip(means, ses)]
    ci_upper = [mean + t_crit * se for mean, se in zip(means, ses)]
    
    # Sort by mean for better visualization
    sorted_indices = sorted(range(len(means)), key=lambda x: means[x], reverse=True)
    
    y_positions = range(len(trust_dimensions))
    for i, idx in enumerate(sorted_indices):
        color = colors[groups[idx]]
        
        # Plot CI
        ax3.plot([ci_lower[idx], ci_upper[idx]], [i, i], 'o-', 
                color=color, linewidth=3, markersize=8, alpha=0.8)
        
        # Add group letter
        ax3.text(ci_upper[idx] + 0.05, i, groups[idx], 
                va='center', fontweight='bold', fontsize=12)
    
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels([trust_dimensions[idx].replace(' ', '\n') for idx in sorted_indices])
    ax3.set_xlabel('Trust Rating (95% CI)', fontsize=12)
    ax3.set_title('95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Effect Sizes Plot (Middle Center)
    ax4 = plt.subplot(3, 3, 5)
    
    # Effect sizes from your results (Cohen's d)
    effect_sizes = [0.448, 0.517, 0.490, 0.565, 0.480, 0.550, 0.305, 0.365]
    pair_labels = ['Clarity vs\nFriendly', 'Clarity vs\nHedging', 'Confidence vs\nFriendly',
                   'Confidence vs\nHedging', 'Citations vs\nFriendly', 'Citations vs\nHedging',
                   'Technical vs\nFriendly', 'Technical vs\nHedging']
    
    # Color code by effect size magnitude
    effect_colors = ['orange' if d >= 0.5 else 'lightcoral' for d in effect_sizes]
    
    bars = ax4.bar(range(len(effect_sizes)), effect_sizes, 
                   color=effect_colors, alpha=0.8, edgecolor='black')
    
    # Add Cohen's d interpretation lines
    ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    
    ax4.set_xticks(range(len(effect_sizes)))
    ax4.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel("Cohen's d", fontsize=12)
    ax4.set_title('Effect Sizes (Significant Pairs Only)', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10)
    
    # Add values on bars
    for bar, d in zip(bars, effect_sizes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{d:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Mean Differences Plot (Middle Right)
    ax5 = plt.subplot(3, 3, 6)
    
    # Create mean differences matrix
    diff_matrix = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            diff_matrix[i, j] = abs(means[i] - means[j])
    
    # Mask diagonal
    mask = np.eye(n_dims, dtype=bool)
    diff_matrix_masked = np.ma.array(diff_matrix, mask=mask)
    
    im = ax5.imshow(diff_matrix_masked, cmap='viridis', vmin=0, vmax=0.6)
    
    # Add text annotations
    for i in range(n_dims):
        for j in range(n_dims):
            if i != j:
                text = ax5.text(j, i, f'{diff_matrix[i, j]:.3f}',
                               ha="center", va="center", color="white", fontweight='bold')
    
    ax5.set_xticks(range(n_dims))
    ax5.set_yticks(range(n_dims))
    ax5.set_xticklabels([dim.replace(' ', '\n') for dim in trust_dimensions], rotation=45)
    ax5.set_yticklabels([dim.replace(' ', '\n') for dim in trust_dimensions])
    ax5.set_title('Mean Differences Matrix\n(Critical = 0.305)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Mean Difference', fontsize=12)
    
    # 6. Summary Statistics Table (Bottom)
    ax6 = plt.subplot(3, 1, 3)
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for i, (dim, mean, std, group) in enumerate(zip(trust_dimensions, means, stds, groups)):
        summary_data.append([dim, f'{mean:.3f}', f'{std:.3f}', group, 
                           f'{len([p for p in sig_pairs if dim in p])}'])
    
    table_data = [['Trust Dimension', 'Mean', 'SD', 'Group', 'Sig. Pairs']] + summary_data
    
    # Create table
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', colWidths=[0.3, 0.15, 0.15, 0.1, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color code by group
    for i in range(1, len(table_data)):
        group_color = colors[groups[i-1]]
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(group_color)
            table[(i, j)].set_text_props(weight='bold', color='white')
    
    # Header formatting
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#333333')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary Statistics by Trust Dimension', 
                  fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('comprehensive_tukey_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive Tukey visualization saved as 'comprehensive_tukey_analysis.png'")
    plt.show()

def create_simple_alternatives():
    """Create simpler, focused visualizations."""
    
    # Alternative 1: Clean Bar Chart with Significance Brackets
    fig, ax = plt.subplots(figsize=(12, 8))
    
    trust_dimensions = ['Source\ncitations', 'Confidence\nscore', 'Clarity', 
                       'Technical\nlanguage', 'Friendly\ntones', 'Hedging\nlanguage']
    means = [3.904, 3.888, 3.864, 3.704, 3.352, 3.304]
    groups = ['A', 'A', 'A', 'A', 'B', 'B']
    
    # Color by group
    colors = ['#2E86AB' if g == 'A' else '#A23B72' for g in groups]
    
    bars = ax.bar(trust_dimensions, means, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add group letters
    for bar, group, mean in zip(bars, groups, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.1,
                group, ha='center', va='bottom', fontweight='bold', 
                fontsize=16, color='black')
    
    # Add significance brackets between groups
    # Group A vs Group B significance
    y_bracket = 4.2
    ax.plot([0, 3], [y_bracket, y_bracket], 'k-', linewidth=2)
    ax.plot([4, 5], [y_bracket-0.2, y_bracket-0.2], 'k-', linewidth=2)
    ax.text(1.5, y_bracket + 0.05, '***', ha='center', fontsize=14, fontweight='bold')
    ax.text(4.5, y_bracket-0.15, 'ns', ha='center', fontsize=12)
    
    ax.set_ylabel('Mean Trust Rating', fontsize=14, fontweight='bold')
    ax.set_title('Trust Dimensions: Tukey HSD Results\nGroup A > Group B (p < 0.001)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylim(0, 4.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(color='#2E86AB', label='Group A (High Trust)'),
                      mpatches.Patch(color='#A23B72', label='Group B (Lower Trust)')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('simple_tukey_bars.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating comprehensive Tukey HSD visualizations...")
    create_tukey_visualizations()
    
    print("\nCreating alternative simple visualization...")
    create_simple_alternatives()
    
    print("\nAll visualizations created successfully!")