import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your dataset (replace this with actual file if needed)
df = pd.read_csv("ExperienceLLMScores.csv")  # Or use pd.read_clipboard() if copying from Excel

# Compute average trust score
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# Ensure categorical order (optional but useful for plotting)
df["Experience"] = df["Experience"].astype("category")
df["LLM"] = df["LLM"].astype("category")

# Plot interaction
plt.figure(figsize=(10, 6))
sns.pointplot(
    data=df,
    x="Experience",
    y="TrustScore",
    hue="LLM",
    dodge=True,
    markers='o',
    capsize=0.1
)
plt.title("Interaction Plot: Trust Score by Experience Level and LLM Persona")
plt.xlabel("Experience Level")
plt.ylabel("Mean Trust Score")
plt.ylim(0, 5)
plt.grid(True)
plt.legend(title="LLM Persona")
plt.tight_layout()
plt.show()
