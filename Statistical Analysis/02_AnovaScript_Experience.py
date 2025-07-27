import pandas as pd
from scipy.stats import f_oneway

# Load or define your dataframe (df)
# If your df contains columns: LLM, Q1, Q2, Q3, Q4, Q5
df = pd.read_csv("ExperienceScores.csv")  # Adjust the path as necessary

# Step 1: Compute average trust score per respondent
df["TrustScore"] = df[["Q1", "Q2", "Q3", "Q4", "Q5"]].mean(axis=1)

# Step 2: Group by LLM and extract scores into list of arrays
anova_data = [group["TrustScore"].values for name, group in df.groupby("Experience")]

# Step 3: Perform one-way ANOVA
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(*anova_data)

# Step 4: Output
print("F-statistic:", round(f_stat, 3))
print("p-value:", round(p_value, 4))
if p_value < 0.05:
    print("✅ Statistically significant differences exist between LLMs.")
else:
    print("❌ No significant differences detected.")
