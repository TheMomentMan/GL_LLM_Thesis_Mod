# Install if not already done: pip install scikit-posthocs
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# Load your data
df = pd.read_csv('Experience_LLM_29thJuly1431.csv')

# Compute trust score
df['Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Kruskal-Wallis Test (non-parametric equivalent of one-way ANOVA)
groups = [df[df['Experience'] == exp]['Score'] for exp in df['Experience'].unique()]
stat, p = kruskal(*groups)
print(f"Kruskal-Wallis Test: H={stat:.4f}, p={p:.4f}")

# Dunn's Test for pairwise comparisons with Bonferroni correction
dunn_results = sp.posthoc_dunn(df, val_col='Score', group_col='Experience', p_adjust='bonferroni')
print("\nDunn's Test Results (p-values):")
print(dunn_results)

# Optional: Heatmap for visualization
plt.figure(figsize=(10, 6))
sns.heatmap(dunn_results, annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'p-value'})
plt.title("Dunn's Test Pairwise Comparison (Experience Groups)")
plt.show()
