import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- STEP 1: Load your dataset ---
df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")

# --- STEP 2: Define and apply custom order ---
custom_order = ["Expert", "Advanced", "Intermediate", "Novice", "Awareness"]
df["Experience"] = pd.Categorical(df["Experience"], categories=custom_order, ordered=True)

# --- STEP 3: Compute average trust score per respondent ---
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# --- STEP 4: Create a pivot table ---
pivot = df.pivot_table(
    index="Experience",
    columns="LLM",
    values="TrustScore",
    aggfunc="mean"
)

# --- STEP 5: Reindex to enforce row order ---
pivot = pivot.reindex(custom_order)

# --- STEP 6: Plot heatmap with strong blue shades ---
plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="Blues",  # Stronger single-color blue palette
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Mean Trust Score'},
    #annot_kws={"size": 10, "color": "black"}
)
plt.title("Heatmap of Mean Trust Scores by Experience and LLM Persona", fontsize=10, pad=10)
plt.xlabel("LLM Persona", fontsize=12)
plt.ylabel("Experience Level", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
