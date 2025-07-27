import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# 1) Load your data
df = pd.read_csv("ExperienceLLMScores.csv")   # or the correct path

# 2) Compute the per-participant average across Q1–Q5
df["Avg"] = df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)

# Sanity check #1: see exactly what Avg looks like
#print("\nAVG value summary:\n", df["Avg"].describe())

# Sanity check #2: make sure it's really the Q1–Q5 mean
#max_error = (df["Avg"] - df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)).abs().max()
#print(f"Maximum per-row mismatch between 'Avg' and Q1–Q5 mean: {max_error}")


# 3) Helper function to run ANOVA, add eta², print & save
def run_anova(formula, df, outname):
    model = smf.ols(formula, data=df).fit()
    aov = anova_lm(model, typ=2).rename_axis("Source")
    # effect size = sum_sq / total SS
    aov["eta_sq"] = aov["sum_sq"] / aov["sum_sq"].sum()
    # round for display
    disp = aov.round(4)
    print(f"\nANOVA: {formula}\n" + "-"*len(formula))
    print(disp.to_string())
    # write out
    disp.to_csv(f"{outname}.csv")

# 4) One-way on LLM
run_anova("Avg ~ C(LLM)", df, "anova_oneway_LLM")

# 5) One-way on Experience
run_anova("Avg ~ C(Experience)", df, "anova_oneway_Experience")

# 6) Two-way (LLM × Experience)
run_anova("Avg ~ C(LLM) * C(Experience)", df, "anova_twoway_LLM_Experience")
