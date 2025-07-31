#!/usr/bin/env python3
"""
Fixed Tukey HSD Post-hoc Analysis for Trust Dimensions
=====================================================
Using statsmodels for proper pairwise comparisons.
"""

import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import studentized_range

def run_tukey_hsd_analysis():
    """
    Run Tukey HSD post-hoc analysis on trust dimensions using manual calculation.
    """
    
    # Load data
    df = pd.read_csv('trust_influencers_July31st.csv')
    
    trust_dimensions = [
        'Clarity',
        'Confidence score', 
        'Source citations',
        'Friendly tones',
        'technical language',
        'Hedging language'
    ]
    
    print("TUKEY HSD POST-HOC ANALYSIS")
    print("=" * 50)
    print("Following up significant ANOVA: F(5, 620) = 13.201, p < 0.001")
    print()
    
    # Calculate descriptive statistics
    group_means = {}
    group_stds = {}
    group_ns = {}
    
    for dim in trust_dimensions:
        data = df[dim].dropna()
        group_means[dim] = data.mean()
        group_stds[dim] = data.std()
        group_ns[dim] = len(data)
    
    print("Group Statistics:")
    print("-" * 40)
    for dim in trust_dimensions:
        print(f"{dim:<20}: M = {group_means[dim]:.3f}, SD = {group_stds[dim]:.3f}, n = {group_ns[dim]}")
    
    # ANOVA parameters (from your previous analysis)
    ms_error = 0.712  # From your ANOVA output
    df_error = 620    # From your ANOVA output
    n_groups = len(trust_dimensions)
    n_per_group = 125  # Same participants rated all dimensions
    
    # Calculate Tukey HSD critical value
    q_critical = studentized_range.ppf(0.95, n_groups, df_error)
    tukey_hsd_critical = q_critical * np.sqrt(ms_error / n_per_group)
    
    print(f"\nTukey HSD Parameters:")
    print(f"q-critical (α = 0.05) = {q_critical:.3f}")
    print(f"MS Error = {ms_error:.3f}")
    print(f"n per group = {n_per_group}")
    print(f"Tukey HSD critical value = {tukey_hsd_critical:.4f}")
    print(f"(Mean differences > {tukey_hsd_critical:.4f} are significant)")
    
    # Perform all pairwise comparisons
    print(f"\nPairwise Comparisons:")
    print("-" * 85)
    print("Group 1               | Group 2               | Mean Diff | Critical | Significant")
    print("-" * 85)
    
    significant_pairs = []
    comparison_results = []
    
    # Get all unique pairs
    pairs = list(combinations(trust_dimensions, 2))
    
    for dim1, dim2 in pairs:
        mean1 = group_means[dim1]
        mean2 = group_means[dim2]
        mean_diff = abs(mean1 - mean2)
        
        # Check significance
        is_significant = mean_diff > tukey_hsd_critical
        
        comparison_results.append({
            'Group1': dim1,
            'Group2': dim2,
            'Mean1': mean1,
            'Mean2': mean2,
            'MeanDiff': mean_diff,
            'Significant': is_significant
        })
        
        if is_significant:
            significant_pairs.append((dim1, dim2, mean_diff))
        
        # Display result
        sig_marker = "***" if is_significant else ""
        print(f"{dim1:<21} | {dim2:<21} | {mean_diff:8.3f} | {tukey_hsd_critical:8.3f} | {sig_marker}")
    
    # Summary of significant differences
    print(f"\nSignificant Pairwise Differences (α = 0.05):")
    print("=" * 55)
    
    if significant_pairs:
        print(f"Found {len(significant_pairs)} significant pairwise differences:")
        for i, (dim1, dim2, diff) in enumerate(significant_pairs, 1):
            mean1 = group_means[dim1]
            mean2 = group_means[dim2]
            higher = dim1 if mean1 > mean2 else dim2
            lower = dim2 if mean1 > mean2 else dim1
            print(f"{i:2d}. {higher} > {lower} (difference: {diff:.3f})")
    else:
        print("No significant pairwise differences found")
    
    # Create homogeneous subsets
    print(f"\nHomogeneous Subsets (Tukey HSD grouping):")
    print("=" * 45)
    
    # Sort dimensions by mean (highest to lowest)
    sorted_dims = sorted(trust_dimensions, key=lambda x: group_means[x], reverse=True)
    
    # Create groups based on significant differences
    groups = []
    remaining_dims = sorted_dims.copy()
    
    while remaining_dims:
        current_group = [remaining_dims[0]]
        remaining_dims.remove(remaining_dims[0])
        
        # Add dimensions that are NOT significantly different from any in current group
        to_remove = []
        for dim in remaining_dims:
            significantly_different = False
            for group_dim in current_group:
                # Check if this pair is significantly different
                for result in comparison_results:
                    if ((result['Group1'] == dim and result['Group2'] == group_dim) or
                        (result['Group1'] == group_dim and result['Group2'] == dim)):
                        if result['Significant']:
                            significantly_different = True
                            break
                if significantly_different:
                    break
            
            if not significantly_different:
                current_group.append(dim)
                to_remove.append(dim)
        
        # Remove dimensions added to current group
        for dim in to_remove:
            remaining_dims.remove(dim)
        
        groups.append(current_group)
    
    # Display homogeneous groups
    for i, group in enumerate(groups, 1):
        group_means_list = [(dim, group_means[dim]) for dim in group]
        group_means_list.sort(key=lambda x: x[1], reverse=True)
        
        group_str = ", ".join([f"{dim} ({mean:.3f})" for dim, mean in group_means_list])
        print(f"Group {chr(64+i)}: {group_str}")
    
    # Calculate effect sizes (Cohen's d) for significant pairs
    if significant_pairs:
        print(f"\nEffect Sizes for Significant Pairs:")
        print("-" * 50)
        for dim1, dim2, diff in significant_pairs:
            # Calculate pooled standard deviation
            std1 = group_stds[dim1]
            std2 = group_stds[dim2]
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            
            # Cohen's d
            cohens_d = diff / pooled_std
            
            # Effect size interpretation
            if cohens_d >= 0.8:
                effect_size = "Large"
            elif cohens_d >= 0.5:
                effect_size = "Medium"
            elif cohens_d >= 0.2:
                effect_size = "Small"
            else:
                effect_size = "Negligible"
            
            mean1 = group_means[dim1]
            mean2 = group_means[dim2]
            higher = dim1 if mean1 > mean2 else dim2
            lower = dim2 if mean1 > mean2 else dim1
            
            print(f"{higher} vs {lower}: Cohen's d = {cohens_d:.3f} ({effect_size})")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # 1. Mean comparison plot with significance groups
    plt.subplot(2, 3, 1)
    sorted_means = [group_means[dim] for dim in sorted_dims]
    sorted_errors = [group_stds[dim]/np.sqrt(group_ns[dim]) for dim in sorted_dims]
    
    # Color code by group
    colors = ['darkgreen', 'green', 'orange', 'red', 'purple', 'brown']
    bar_colors = []
    
    for dim in sorted_dims:
        for i, group in enumerate(groups):
            if dim in group:
                bar_colors.append(colors[i % len(colors)])
                break
    
    bars = plt.bar(range(len(sorted_dims)), sorted_means, 
                   yerr=sorted_errors, capsize=5, alpha=0.7, color=bar_colors)
    
    plt.xticks(range(len(sorted_dims)), 
               [dim.replace(' ', '\n') for dim in sorted_dims], 
               rotation=0, ha='center')
    plt.ylabel('Mean Rating')
    plt.title('Trust Dimensions with Tukey HSD Groups')
    
    # Add group letters above bars
    for i, (bar, dim) in enumerate(zip(bars, sorted_dims)):
        # Find which group this dimension belongs to
        for j, group in enumerate(groups):
            if dim in group:
                group_letter = chr(65+j)  # A, B, C, etc.
                plt.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + sorted_errors[i] + 0.05,
                        group_letter, ha='center', va='bottom', 
                        fontweight='bold', fontsize=14, color='black')
                break
    
    # Add horizontal line for critical difference
    plt.axhline(y=max(sorted_means) - tukey_hsd_critical, 
                color='red', linestyle='--', alpha=0.5, 
                label=f'Critical Diff = {tukey_hsd_critical:.3f}')
    plt.legend()
    
    # 2. Pairwise comparison matrix
    plt.subplot(2, 3, 2)
    
    # Create significance matrix
    sig_matrix = np.zeros((len(trust_dimensions), len(trust_dimensions)))
    
    for result in comparison_results:
        i = trust_dimensions.index(result['Group1'])
        j = trust_dimensions.index(result['Group2'])
        sig_value = 1 if result['Significant'] else 0
        sig_matrix[i, j] = sig_value
        sig_matrix[j, i] = sig_value
    
    # Set diagonal to NaN for better visualization
    np.fill_diagonal(sig_matrix, np.nan)
    
    # Plot heatmap
    sns.heatmap(sig_matrix, 
                xticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                yticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                annot=True, fmt='.0f', cmap='RdYlBu_r', center=0.5,
                cbar_kws={'label': 'Significantly Different'})
    plt.title('Pairwise Significance Matrix\n(1 = Significant, 0 = Not Significant)')
    
    # 3. Mean differences heatmap
    plt.subplot(2, 3, 3)
    
    # Create mean differences matrix
    diff_matrix = np.zeros((len(trust_dimensions), len(trust_dimensions)))
    
    for i, dim1 in enumerate(trust_dimensions):
        for j, dim2 in enumerate(trust_dimensions):
            if i != j:
                diff_matrix[i, j] = abs(group_means[dim1] - group_means[dim2])
    
    np.fill_diagonal(diff_matrix, np.nan)
    
    sns.heatmap(diff_matrix,
                xticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                yticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'Mean Differences\n(Critical = {tukey_hsd_critical:.3f})')
    
    # 4. Effect sizes heatmap
    plt.subplot(2, 3, 4)
    
    # Calculate Cohen's d for all pairs
    cohens_d_matrix = np.zeros((len(trust_dimensions), len(trust_dimensions)))
    
    for i, dim1 in enumerate(trust_dimensions):
        for j, dim2 in enumerate(trust_dimensions):
            if i != j:
                mean_diff = abs(group_means[dim1] - group_means[dim2])
                std1 = group_stds[dim1]
                std2 = group_stds[dim2]
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = mean_diff / pooled_std
                cohens_d_matrix[i, j] = cohens_d
    
    np.fill_diagonal(cohens_d_matrix, np.nan)
    
    sns.heatmap(cohens_d_matrix,
                xticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                yticklabels=[dim.replace(' ', '\n') for dim in trust_dimensions],
                annot=True, fmt='.2f', cmap='plasma')
    plt.title('Effect Sizes (Cohen\'s d)\nBetween Trust Dimensions')
    
    # 5. Forest plot of means with confidence intervals
    plt.subplot(2, 3, 5)
    
    # Calculate 95% confidence intervals
    from scipy.stats import t
    alpha = 0.05
    df_t = group_ns[trust_dimensions[0]] - 1  # n-1 for each group
    t_critical = t.ppf(1 - alpha/2, df_t)
    
    y_positions = range(len(sorted_dims))
    
    for i, dim in enumerate(sorted_dims):
        mean = group_means[dim]
        std_err = group_stds[dim] / np.sqrt(group_ns[dim])
        ci_lower = mean - t_critical * std_err
        ci_upper = mean + t_critical * std_err
        
        # Plot confidence interval
        plt.plot([ci_lower, ci_upper], [i, i], 'b-', linewidth=2)
        plt.plot([ci_lower, ci_lower], [i-0.1, i+0.1], 'b-', linewidth=2)
        plt.plot([ci_upper, ci_upper], [i-0.1, i+0.1], 'b-', linewidth=2)
        
        # Plot mean
        plt.plot(mean, i, 'ro', markersize=8)
        
        # Add group letter
        for j, group in enumerate(groups):
            if dim in group:
                group_letter = chr(65+j)
                plt.text(mean + 0.05, i, group_letter, 
                        va='center', fontweight='bold', fontsize=12)
                break
    
    plt.yticks(y_positions, [dim.replace(' ', '\n') for dim in sorted_dims])
    plt.xlabel('Rating')
    plt.title('95% Confidence Intervals\nwith Tukey Groups')
    plt.grid(True, alpha=0.3)
    
    # 6. Summary table
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Create summary text
    summary_text = f"""
TUKEY HSD SUMMARY

Significant Comparisons: {len(significant_pairs)} of {len(pairs)}
Critical Value: {tukey_hsd_critical:.4f}
Homogeneous Groups: {len(groups)}

GROUP ASSIGNMENTS:
"""
    
    for i, group in enumerate(groups, 1):
        group_range = f"{min([group_means[dim] for dim in group]):.3f}-{max([group_means[dim] for dim in group]):.3f}"
        summary_text += f"Group {chr(64+i)}: {group_range}\n"
        for dim in group:
            summary_text += f"  • {dim}\n"
        summary_text += "\n"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('tukey_hsd_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nTukey HSD visualization saved as 'tukey_hsd_analysis.png'")
    plt.show()
    
    # Summary for thesis
    print(f"\n" + "="*70)
    print("SUMMARY FOR THESIS")
    print("="*70)
    print(f"Tukey HSD post-hoc analysis revealed:")
    print(f"• {len(significant_pairs)} significant pairwise differences out of {len(pairs)} comparisons")
    print(f"• {len(groups)} homogeneous subsets identified")
    print(f"• Critical difference for significance: {tukey_hsd_critical:.4f}")
    
    if len(groups) > 1:
        print(f"\nTrust dimension hierarchy:")
        for i, group in enumerate(groups, 1):
            group_means_in_group = [group_means[dim] for dim in group]
            group_range = f"{min(group_means_in_group):.3f}-{max(group_means_in_group):.3f}"
            print(f"  Tier {i} (Group {chr(64+i)}): {', '.join(group)} (M = {group_range})")
    
    return {
        'significant_pairs': significant_pairs,
        'homogeneous_groups': groups,
        'critical_value': tukey_hsd_critical,
        'comparison_results': comparison_results,
        'group_means': group_means
    }

if __name__ == "__main__":
    results = run_tukey_hsd_analysis()