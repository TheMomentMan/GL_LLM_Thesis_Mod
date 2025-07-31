import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Step 1: Load your Excel dataset
df = pd.read_excel("/All_LLM_Data_everything.xlsx")

# Step 2: Reshape Gen3–Gen7 into long form
gen_cols = ["Gen3", "Gen4", "Gen5", "Gen6", "Gen7"]
long_df = df[gen_cols].melt(var_name="Factor", value_name="TrustScore").dropna()

# Step 3: Run Tukey's HSD
tukey = pairwise_tukeyhsd(endog=long_df["TrustScore"],
                          groups=long_df["Factor"],
                          alpha=0.05)

# Step 4: Create results DataFrame
results_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
results_df.columns = ["Factor1", "Factor2", "Mean Diff", "p-adjusted", "Lower CI", "Upper CI", "Reject Ho"]
results_df["Critical Value"] = 0.305  # Optional: to match your table style
results_df = results_df[["Factor1", "Factor2", "Mean Diff", "Critical Value", "p-adjusted", "Lower CI", "Upper CI", "Reject Ho"]]

# Step 5: Build symmetric p-value matrix for heatmap
pivot_pvals = results_df.pivot(index="Factor1", columns="Factor2", values="p-adjusted")
pval_matrix = pivot_pvals.copy()

for i in pivot_pvals.index:
    for j in pivot_pvals.columns:
        if pd.notna(pivot_pvals.at[i, j]):
            pval_matrix.at[j, i] = pivot_pvals.at[i, j]

# Fill diagonal and missing values with 1 (non-significant)
for label in gen_cols:
    pval_matrix.at[label, label] = 1.0
pval_matrix.fillna(1.0, inplace=True)

# Step 6: Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pval_matrix.astype(float), annot=True, fmt=".3f", cmap="coolwarm_r", cbar_kws={'label': 'p-adjusted'})
plt.title("Tukey HSD Pairwise p-values (Gen3–Gen7)")
plt.tight_layout()
plt.show()

# Step 7: Optional — export table
# results_df.to_excel("Tukey_HSD_Gen3toGen7.xlsx", index=False)

# Step 8: Show results in console
print(results_df.round(4))
