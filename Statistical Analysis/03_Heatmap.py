import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- STEP 1: Load your dataset ---
df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")  # Update path if needed

# --- STEP 2: Compute average trust score per respondent ---
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# --- STEP 3: Create a pivot table of group means ---
pivot = df.pivot_table(
    index="Experience",  # Rows = experience levels
    columns="LLM",       # Columns = LLM personas
    values="TrustScore", # Values = average trust score
    aggfunc="mean"       # Aggregate function
)

# --- STEP 4: Plot the heatmap ---
plt.figure(figsize=(8, 5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)
plt.title("Heatmap of Mean Trust Scores by Experience and LLM")
plt.xlabel("LLM Persona")
plt.ylabel("Experience Level")
plt.tight_layout()
plt.show()
