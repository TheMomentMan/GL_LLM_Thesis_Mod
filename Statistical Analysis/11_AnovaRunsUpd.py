import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# 1) Load your data — but *only* grab the seven columns we actually need
df = pd.read_csv("Experience_LLM_26thJuly1224.csv", usecols=["LLM","Experience","Q1","Q2","Q3","Q4","Q5"])

# 2) Force our grouping vars to be categorical (so Patsy won’t get confused)
df["LLM"] = df["LLM"].astype("category")
df["Experience"] = df["Experience"].astype("category")

# 3) Compute the per-participant average across Q1–Q5
df["Avg"] = df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)


def run_anova(formula, df, outname):
    """Fits the formula, runs Type-II ANOVA, adds eta², prints & writes CSV."""
    model = smf.ols(formula, data=df).fit()
    aov = anova_lm(model, typ=2).rename_axis("Source")
    aov["eta_sq"] = aov["sum_sq"] / aov["sum_sq"].sum()
    disp = aov.round(4)
    print(f"\nANOVA: {formula}\n" + "-"*len(formula))
    print(disp.to_string())
    disp.to_csv(f"{outname}.csv")


# 4) One-way on LLM
run_anova("Avg ~ C(LLM)", df,   "anova_oneway_LLM")

# 5) One-way on Experience
run_anova("Avg ~ C(Experience)", df, "anova_oneway_Experience")

# 6) Two-way with interaction
run_anova("Avg ~ C(LLM) * C(Experience)", df, "anova_twoway_LLM_Experience")
