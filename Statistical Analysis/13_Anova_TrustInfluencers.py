#!/usr/bin/env python3
"""
Trust Influencers ANOVA Analysis
================================
Analyzes whether differences in trust ratings across 6 dimensions are statistically significant.

Trust Dimensions:
1. Clarity: Clear explanations of AI's reasoning increase trust
2. Confidence score: Confidence scores/probability estimates increase trust  
3. Source citations: Trust increases when sources are provided
4. Friendly tones: Friendly tone makes AI feel more trustworthy
5. Technical language: Trust increases with appropriate technical terms
6. Hedging language: Distrust AI responses that sound overly confident

This is a repeated measures ANOVA since each participant rated all 6 dimensions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f
import matplotlib.pyplot as plt
import seaborn as sns

def run_trust_influencers_anova():
    """Run complete ANOVA analysis on trust influencers data."""
    
    # Load data
    try:
        df = pd.read_csv('trust_influencers_july29th.csv')
        print("Trust Influencers ANOVA Analysis")
        print("=" * 50)
        print(f"Data loaded: {len(df)} participants")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("Error: trust_influencers_july29th.csv not found")
        return
    
    # Define trust dimensions and their full questions
    trust_dimensions = [
        'Clarity',
        'Confidence score', 
        'Source citations',
        'Friendly tones',
        'technical language',
        'Hedging language'
    ]
    
    questions = {
        'Clarity': 'Clear explanations of AI\'s reasoning increase trust',
        'Confidence score': 'Confidence scores/probability estimates increase trust',
        'Source citations': 'Trust increases when sources are provided',
        'Friendly tones': 'Friendly tone makes AI feel more trustworthy',
        'technical language': 'Trust increases with appropriate technical terms',
        'Hedging language': 'Distrust overly confident responses'
    }
    
    print("\nTrust Dimension Questions:")
    print("-" * 30)
    for dim, question in questions.items():
        print(f"{dim}: {question}")
    
    # Check for missing data and prepare for analysis
    complete_cases = df[trust_dimensions].dropna()
    n_complete = len(complete_cases)
    n_missing = len(df) - n_complete
    
    print(f"\nData Quality:")
    print(f"Complete cases: {n_complete}")
    print(f"Cases with missing data: {n_missing}")
    
    if n_missing > 0:
        print("Using complete cases only for analysis")
        df_analysis = df[df[trust_dimensions].notna().all(axis=1)].copy()
    else:
        df_analysis = df.copy()
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics (n={len(df_analysis)}):")
    print("=" * 50)
    
    desc_stats = df_analysis[trust_dimensions].describe()
    print(desc_stats.round(3))
    
    # Calculate means for ranking
    means = df_analysis[trust_dimensions].mean().sort_values(ascending=False)
    stds = df_analysis[trust_dimensions].std()
    
    print(f"\nTrust Dimension Ranking:")
    print("-" * 25)
    for i, (dimension, mean_val) in enumerate(means.items(), 1):
        std_val = stds[dimension]
        print(f"{i}. {dimension}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Prepare data for repeated measures ANOVA
    # Reshape to long format
    df_long = pd.melt(df_analysis.reset_index(), 
                      id_vars=['index'], 
                      value_vars=trust_dimensions,
                      var_name='Dimension', 
                      value_name='Rating')
    df_long['Participant'] = df_long['index']
    
    # Manual Repeated Measures ANOVA calculation
    print(f"\nRepeated Measures ANOVA Calculations:")
    print("=" * 40)
    
    # Basic parameters
    n_participants = len(df_analysis)
    n_dimensions = len(trust_dimensions)
    n_total = n_participants * n_dimensions
    
    print(f"Number of participants: {n_participants}")
    print(f"Number of dimensions: {n_dimensions}")
    print(f"Total observations: {n_total}")
    
    # Calculate means
    grand_mean = df_long['Rating'].mean()
    participant_means = df_long.groupby('Participant')['Rating'].mean()
    dimension_means = df_long.groupby('Dimension')['Rating'].mean()
    
    print(f"Grand mean: {grand_mean:.3f}")
    
    # Sum of Squares calculations
    
    # Total Sum of Squares
    SS_total = ((df_long['Rating'] - grand_mean) ** 2).sum()
    
    # Between-Subjects Sum of Squares
    SS_between_subjects = n_dimensions * ((participant_means - grand_mean) ** 2).sum()
    
    # Within-Subjects Sum of Squares
    SS_within_subjects = SS_total - SS_between_subjects
    
    # Between-Dimensions Sum of Squares (Treatment effect)
    SS_between_dimensions = n_participants * ((dimension_means - grand_mean) ** 2).sum()
    
    # Error Sum of Squares
    SS_error = SS_within_subjects - SS_between_dimensions
    
    # Degrees of freedom
    df_total = n_total - 1
    df_between_subjects = n_participants - 1
    df_within_subjects = n_total - n_participants
    df_between_dimensions = n_dimensions - 1
    df_error = df_within_subjects - df_between_dimensions
    
    # Mean squares
    MS_between_dimensions = SS_between_dimensions / df_between_dimensions
    MS_error = SS_error / df_error
    
    # F-statistic
    F_stat = MS_between_dimensions / MS_error
    
    # P-value
    p_value = 1 - f.cdf(F_stat, df_between_dimensions, df_error)
    
    # F-critical values
    F_crit_05 = f.ppf(0.95, df_between_dimensions, df_error)
    F_crit_01 = f.ppf(0.99, df_between_dimensions, df_error)
    
    # Effect size (eta-squared)
    eta_squared = SS_between_dimensions / SS_total
    
    # Print calculations
    print(f"\nSum of Squares:")
    print(f"SS Total = {SS_total:.3f}")
    print(f"SS Between Subjects = {SS_between_subjects:.3f}")
    print(f"SS Within Subjects = {SS_within_subjects:.3f}")
    print(f"SS Between Dimensions = {SS_between_dimensions:.3f}")
    print(f"SS Error = {SS_error:.3f}")
    
    print(f"\nDegrees of Freedom:")
    print(f"df Total = {df_total}")
    print(f"df Between Subjects = {df_between_subjects}")
    print(f"df Within Subjects = {df_within_subjects}")
    print(f"df Between Dimensions = {df_between_dimensions}")
    print(f"df Error = {df_error}")
    
    print(f"\nMean Squares:")
    print(f"MS Between Dimensions = {MS_between_dimensions:.3f}")
    print(f"MS Error = {MS_error:.3f}")
    
    print(f"\nF-Statistic and Significance:")
    print(f"F = {F_stat:.3f}")
    print(f"p-value = {p_value:.6f}")
    print(f"F-critical (α = 0.05) = {F_crit_05:.3f}")
    print(f"F-critical (α = 0.01) = {F_crit_01:.3f}")
    print(f"Eta-squared (η²) = {eta_squared:.4f}")
    
    # Standard ANOVA Table
    print(f"\nRepeated Measures ANOVA Table:")
    print("=" * 70)
    print("Source                | SS        | df  | MS        | F      | p-value")
    print("-" * 70)
    print(f"Between Dimensions    | {SS_between_dimensions:9.3f} | {df_between_dimensions:3d} | {MS_between_dimensions:9.3f} | {F_stat:6.3f} | {p_value:.6f}")
    print(f"Error (Within)        | {SS_error:9.3f} | {df_error:3d} | {MS_error:9.3f} |        |")
    print(f"Between Subjects      | {SS_between_subjects:9.3f} | {df_between_subjects:3d} |           |        |")
    print(f"Total                 | {SS_total:9.3f} | {df_total:3d} |           |        |")
    
    # Effect size interpretation
    if eta_squared >= 0.14:
        effect_size = "Large"
    elif eta_squared >= 0.06:
        effect_size = "Medium"
    elif eta_squared >= 0.01:
        effect_size = "Small"
    else:
        effect_size = "Negligible"
    
    # Statistical conclusion
    print(f"\nStatistical Conclusion:")
    print("=" * 25)
    
    if p_value < 0.05:
        significance = "SIGNIFICANT"
        print(f"✓ {significance} RESULT")
        print(f"The differences between trust dimensions are statistically significant")
        print(f"F({df_between_dimensions}, {df_error}) = {F_stat:.3f}, p = {p_value:.6f}")
        print(f"Effect size: η² = {eta_squared:.4f} ({effect_size.lower()} effect)")
        print(f"Conclusion: Trust influencer ratings are NOT equal across dimensions")
        
        # Post-hoc interpretation
        print(f"\nPost-hoc Interpretation:")
        print("• Participants distinguish between different trust factors")
        print("• Some dimensions are more important than others for trust")
        print("• The highest-rated dimensions should be prioritized in AI design")
        
    else:
        significance = "NOT SIGNIFICANT"
        print(f"✗ {significance}")
        print(f"No significant differences between trust dimensions")
        print(f"F({df_between_dimensions}, {df_error}) = {F_stat:.3f}, p = {p_value:.6f}")
        print(f"Conclusion: Trust ratings are similar across all dimensions")
        
        print(f"\nInterpretation:")
        print("• All trust dimensions are rated similarly by participants")
        print("• No single factor stands out as more important")
        print("• Participants value all aspects equally for trust building")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Box plot of ratings by dimension
    plt.subplot(2, 2, 1)
    df_analysis[trust_dimensions].boxplot(ax=plt.gca(), rot=45)
    plt.title('Trust Ratings by Dimension')
    plt.ylabel('Rating')
    plt.tight_layout()
    
    # Mean ratings with error bars
    plt.subplot(2, 2, 2)
    means_plot = df_analysis[trust_dimensions].mean()
    errors_plot = df_analysis[trust_dimensions].std() / np.sqrt(len(df_analysis))
    
    bars = plt.bar(range(len(means_plot)), means_plot.values, 
                   yerr=errors_plot.values, capsize=5, alpha=0.7)
    plt.xticks(range(len(means_plot)), means_plot.index, rotation=45, ha='right')
    plt.ylabel('Mean Rating')
    plt.title('Mean Trust Ratings (±SEM)')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means_plot.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors_plot.iloc[i],
                f'{mean_val:.2f}', ha='center', va='bottom')
    
    # Correlation matrix
    plt.subplot(2, 2, 3)
    corr_matrix = df_analysis[trust_dimensions].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Trust Dimensions Correlation Matrix')
    
    # Distribution of ratings
    plt.subplot(2, 2, 4)
    df_long_plot = df_analysis[trust_dimensions].melt(var_name='Dimension', value_name='Rating')
    sns.violinplot(data=df_long_plot, x='Dimension', y='Rating')
    plt.xticks(rotation=45, ha='right')
    plt.title('Rating Distributions by Dimension')
    
    plt.tight_layout()
    plt.savefig('trust_influencers_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'trust_influencers_analysis.png'")
    plt.show()
    
    # Return results for further analysis
    results = {
        'F_statistic': F_stat,
        'p_value': p_value,
        'F_critical_05': F_crit_05,
        'F_critical_01': F_crit_01,
        'eta_squared': eta_squared,
        'effect_size': effect_size,
        'significance': significance,
        'SS_between_dimensions': SS_between_dimensions,
        'SS_error': SS_error,
        'SS_total': SS_total,
        'df_between_dimensions': df_between_dimensions,
        'df_error': df_error,
        'MS_between_dimensions': MS_between_dimensions,
        'MS_error': MS_error,
        'dimension_means': dimension_means.sort_values(ascending=False),
        'dimension_ranking': means
    }
    
    return results

if __name__ == "__main__":
    results = run_trust_influencers_anova()
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Key finding: {results['significance']}")
    print(f"F-statistic: {results['F_statistic']:.3f}")
    print(f"p-value: {results['p_value']:.6f}")
    print(f"Effect size: {results['effect_size']} (η² = {results['eta_squared']:.4f})")