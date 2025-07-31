from scipy.stats import levene
import pandas as pd

df = pd.read_csv('Experience_LLM_29thJuly1431.csv')
# Calculate trust scores (average of Q1-Q5)
df['Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Unpack scores per group
groups = [df[df['Experience'] == g]['Score'] for g in df['Experience'].unique()]
stat, p = levene(*groups)
print(f"Levene's test: W={stat:.3f}, p={p:.3f} {'(equal variances)' if p > 0.05 else '(unequal variances)'}")
