import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1) Load & prepare
df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")
df["Avg"] = df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)

# 2) Run Tukey HSD for the factor “LLM” (you can swap in “Experience” etc.)
tukey = pairwise_tukeyhsd(
    endog=df["Avg"],       # the dependent variable
    groups=df["Experience"],      # the factor
    alpha=0.05             # 95% confidence
)

# 3) Print the familiar table‐in‐a‐grid
print(tukey.summary())

# 4) Extract into a DataFrame and save to CSV
#    tukey._results_table.data is a list-of-lists: first row is header, then rows
tukey_rows = tukey._results_table.data
tukey_df   = pd.DataFrame(tukey_rows[1:], columns=tukey_rows[0])
tukey_df.to_csv("tukey_LLM.csv", index=False)

print("\n→ Tukey table also written to tukey_LLM.csv")
