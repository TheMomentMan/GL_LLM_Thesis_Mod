import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Step 1: Load your dataset (replace this with actual file if needed)
df = pd.read_csv("ExperienceLLMScores.csv")  # Or use pd.read_clipboard() if copying from Excel

df["LLM"] = df["LLM"].str.strip().str.upper()
df["Experience"] = df["Experience"].str.strip().str.title()

# Step 2: Compute average trust score
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# Step 3: Two-Way ANOVA
model = ols("TrustScore ~ C(LLM) + C(Experience) + C(LLM):C(Experience)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Step 4: View results
print(anova_table)
