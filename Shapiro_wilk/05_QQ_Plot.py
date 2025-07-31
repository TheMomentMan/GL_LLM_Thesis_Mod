import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Experience_LLM_29thJuly1431.csv')
# Calculate trust scores (average of Q1-Q5)
df['Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)
for group in df['LLM'].unique():
    sm.qqplot(df[df['LLM'] == group]['Score'], line='s')
    plt.title(f'QQ Plot - {group}')
    plt.show()
