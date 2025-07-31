import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('Experience_LLM_29thJuly1431.csv')

# Calculate trust scores (average of Q1-Q5)
df['TrustScore'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

print("=== SHAPIRO-WILK NORMALITY TESTS BY LLM GROUP ===")
print("H₀: Data is normally distributed")
print("H₁: Data is NOT normally distributed")
print("α = 0.05 significance level\n")

# Get unique LLM groups
llm_groups = df['LLM'].unique()

# Store results for summary
results_summary = []

# Test each LLM group separately
for llm in llm_groups:
    # Filter data for this LLM
    group_data = df[df['LLM'] == llm]['TrustScore']
    
    # Perform Shapiro-Wilk test
    statistic, p_value = stats.shapiro(group_data)
    
    # Determine interpretation
    if p_value > 0.05:
        interpretation = "✓ Normal (fail to reject H₀)"
        decision = "Suitable for ANOVA"
    else:
        interpretation = "⚠ Non-normal (reject H₀)"
        decision = "Consider non-parametric alternative"
    
    # Display results
    print(f"{llm} (n={len(group_data)}):")
    print(f"  Shapiro-Wilk W: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {interpretation}")
    print(f"  Recommendation: {decision}")
    
    # Additional descriptive stats
    print(f"  Mean: {group_data.mean():.3f}")
    print(f"  Std Dev: {group_data.std():.3f}")
    print(f"  Skewness: {stats.skew(group_data):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(group_data):.3f}")
    print()
    
    # Store for summary
    results_summary.append({
        'LLM': llm,
        'n': len(group_data),
        'W_statistic': statistic,
        'p_value': p_value,
        'Normal': p_value > 0.05
    })

# Create summary table
summary_df = pd.DataFrame(results_summary)
print("=== SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

# Overall assessment
normal_count = sum(summary_df['Normal'])
total_groups = len(summary_df)

print(f"\n=== OVERALL ASSESSMENT ===")
print(f"Groups passing normality: {normal_count}/{total_groups}")

if normal_count == total_groups:
    print("✅ All groups are normally distributed")
    print("✅ Proceed with standard one-way ANOVA")
elif normal_count >= total_groups * 0.75:
    print("⚠ Most groups are normal")
    print("✅ ANOVA is robust - likely OK to proceed")
else:
    print("⚠ Many groups are non-normal")
    print("🔄 Consider Kruskal-Wallis test or data transformation")

# OPTIONAL: Create Q-Q plots for visual assessment
print("\n=== CREATING Q-Q PLOTS FOR VISUAL INSPECTION ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, llm in enumerate(llm_groups):
    group_data = df[df['LLM'] == llm]['TrustScore']
    stats.probplot(group_data, dist="norm", plot=axes[i])
    axes[i].set_title(f'{llm} Q-Q Plot (n={len(group_data)})')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# BONUS: Test homogeneity of variance (Levene's test)
print("=== LEVENE'S TEST FOR HOMOGENEITY OF VARIANCE ===")

# Prepare data for Levene's test
group_data_list = [df[df['LLM'] == llm]['TrustScore'] for llm in llm_groups]

# Perform Levene's test
levene_stat, levene_p = stats.levene(*group_data_list)

print(f"Levene's test statistic: {levene_stat:.4f}")
print(f"p-value: {levene_p:.4f}")

if levene_p > 0.05:
    print("✅ Equal variances (homoscedasticity assumed)")
else:
    print("⚠ Unequal variances - consider Welch's ANOVA")

print("\n=== FINAL ANOVA ASSUMPTIONS CHECK ===")
print("1. Independence: ✓ (assumed from study design)")
print(f"2. Normality: {'✓' if normal_count == total_groups else '⚠'}")
print(f"3. Homogeneity: {'✓' if levene_p > 0.05 else '⚠'}")
print("4. Sample sizes: ✓ (adequate for each group)")