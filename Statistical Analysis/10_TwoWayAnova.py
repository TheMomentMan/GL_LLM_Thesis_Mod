import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# 1) Load your data
df = pd.read_csv("ExperienceLLMScores.csv")  # or .csv

# 2) Create the Avg column
questions = ["Q1","Q2","Q3","Q4","Q5"]
df["Avg"] = df[questions].mean(axis=1)

# 3) Helper to run ANOVA and print a table
def run_anova(formula, df, name="anova"):
    print(f"\nANOVA: {formula}\n" + "-"*40)
    model = smf.ols(formula, data=df).fit()
    aov = anova_lm(model, typ=2).rename_axis("Source")
    print(aov.round(4).to_string())
    # If you’d rather export to CSV:
    aov.round(4).to_csv(f"{name}.csv")
    return aov

# 4) One-way on LLM persona
run_anova("Avg ~ C(LLM)", df,   name="anova_llm")

# 5) One-way on Experience
run_anova("Avg ~ C(Experience)", df, name="anova_exp")

# 6) Two-way with interaction
run_anova("Avg ~ C(LLM) * C(Experience)", df, name="anova_2way")
