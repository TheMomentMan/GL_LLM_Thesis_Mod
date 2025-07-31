import pandas as pd
import numpy as np
from itertools import combinations
import scipy.stats as stats
import math

# Load your data
df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")
df['TrustScore'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Helper function to compute Cohen's d
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)
    d = (np.mean(x) - np.mean(y)) / pooled_std
    return d

# Effect sizes and confidence intervals
results = []
for group1, group2 in combinations(df['LLM'].unique(), 2):
    x = df[df['LLM'] == group1]['TrustScore']
    y = df[df['LLM'] == group2]['TrustScore']
    
    mean_diff = np.mean(x) - np.mean(y)
    d = cohens_d(x, y)
    
    # 95% CI for Cohen's d using normal approximation
    nx, ny = len(x), len(y)
    se_d = np.sqrt((nx + ny) / (nx * ny) + (d**2 / (2 * (nx + ny))))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d
    
    results.append({
        'Group 1': group1,
        'Group 2': group2,
        'Mean Difference': round(mean_diff, 3),
        'Cohen\'s d': round(d, 3),
        '95% CI Lower': round(ci_lower, 3),
        '95% CI Upper': round(ci_upper, 3)
    })

effect_df = pd.DataFrame(results)
print(effect_df)
