import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
df = pd.read_csv('Experience_LLM_30thJuly_1349.csv')
custom_order = ["Awareness", "Novice", "Intermediate", "Advanced", "Expert"]
custom_orderL = ["LLM_A", "LLM_B", "LLM_C", "LLM_D"]
df['Experience'] = pd.Categorical(df['Experience'], categories=custom_order, ordered=True)

# Calculate average trust score across Q1–Q5
df['Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Get unique LLM groups and determine layout
#llms = df['LLM'].unique()
llms = [llm for llm in custom_orderL if llm in df['LLM'].unique()]
#llms = [exp for exp in custom_order if exp in df['Experience'].unique()]
n = len(llms)
cols = 2
rows = (n + 1) // cols

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axes = axes.flatten()  # To simplify indexing even if grid is 1 row

# Generate QQ plot for each LLM group
for i, group in enumerate(llms):
    ax = axes[i]
    sm.qqplot(df[df['LLM'] == group]['Score'], line='s', ax=ax)
    ax.set_title(f'QQ Plot - {group}')

# Remove any extra subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
