import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")

# Compute average trust score
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# Get unique LLMs
llms = df["LLM"].unique()

# Prepare pairwise comparisons
results = []
for g1, g2 in itertools.combinations(llms, 2):
    scores1 = df[df["LLM"] == g1]["TrustScore"]
    scores2 = df[df["LLM"] == g2]["TrustScore"]

    mean1, mean2 = scores1.mean(), scores2.mean()
    diff = mean1 - mean2

    n1, n2 = len(scores1), len(scores2)
    s1, s2 = scores1.std(ddof=1), scores2.std(ddof=1)

    # Pooled SD
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2))
    cohen_d = diff / s_pooled

    # Standard error for diff
    se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
    t_crit = stats.t.ppf(0.975, n1 + n2 - 2)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    results.append({
        "Group 1": g1,
        "Group 2": g2,
        "Mean Difference": round(diff, 3),
        "Cohen's d": round(cohen_d, 3),
        "95% CI Lower": round(ci_lower, 3),
        "95% CI Upper": round(ci_upper, 3)
    })

# Create DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Save to CSV
results_df.to_csv("LLM_pairwise_cohensd.csv", index=False)

# ---- Optional: Forest Plot ----
plt.figure(figsize=(8, 6))
y_pos = np.arange(len(results_df))[::-1]

plt.errorbar(results_df["Mean Difference"], y_pos,
             xerr=[results_df["Mean Difference"] - results_df["95% CI Lower"],
                   results_df["95% CI Upper"] - results_df["Mean Difference"]],
             fmt='o', color='blue', ecolor='gray', capsize=5)

plt.yticks(y_pos, results_df["Group 1"] + " vs " + results_df["Group 2"])
plt.axvline(x=0, color='red', linestyle='--')
plt.xlabel("Mean Difference (with 95% CI)")
plt.title("Pairwise LLM Trust Score Comparisons")
plt.tight_layout()
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.show()
