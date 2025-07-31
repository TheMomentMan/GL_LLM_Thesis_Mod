from scipy.stats import shapiro
import pandas as pd


df = pd.read_csv('Experience_LLM_30thJuly_1349.csv')
# Calculate trust scores (average of Q1-Q5)
df['TrustScore'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Assume df is your DataFrame, and you want to test "Score" across "Group"
for group in df['LLM'].unique():
    stat, p = shapiro(df[df['LLM'] == group]['TrustScore'])
    print(f'{group}: W={stat:.3f}, p={p:.3f} {"(normal)" if p > 0.05 else "(non-normal)"}')
