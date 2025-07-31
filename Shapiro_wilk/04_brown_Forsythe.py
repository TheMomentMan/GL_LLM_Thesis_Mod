from scipy.stats import levene
import pandas as pd

df = pd.read_csv('Exper30thJulyy24.csv')
# Calculate trust scores (average of Q1-Q5)
df['Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)
# Brown-Forsythe test (center='median')
groups = [df[df['Experience'] == g]['Score'] for g in df['Experience'].unique()]
stat, p = levene(*groups, center='median')
print(f"Brown-Forsythe test: W={stat:.3f}, p={p:.3f}")
