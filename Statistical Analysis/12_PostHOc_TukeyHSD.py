import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1) Load & prep
df = pd.read_csv("ExperienceLLMScores.csv")
df["Avg"] = df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)

# 2) Tukey HSD for LLM persona
tukey_llm = pairwise_tukeyhsd(
    endog=df["Avg"],
    groups=df["LLM"],
    alpha=0.05
)
print("\nTukey HSD – LLM persona\n", tukey_llm)

# 3) Tukey HSD for Experience
tukey_exp = pairwise_tukeyhsd(
    endog=df["Avg"],
    groups=df["Experience"],
    alpha=0.05
)
print("\nTukey HSD – Experience\n", tukey_exp)

# 4) (Optional) Simple main effects of LLM *within* each Experience level
for lvl in df["Experience"].unique():
    sub = df[df["Experience"] == lvl]
    print(f"\nTukey HSD – LLM within Experience = {lvl}")
    print(pairwise_tukeyhsd(endog=sub["Avg"], groups=sub["LLM"], alpha=0.05))
