import pandas as pd

df = pd.read_csv("ExperienceLLMScores.csv")

# 1) Show the first few rows & column types
print(df.head(5))
print(df.dtypes, "\n")

# 2) Confirm your five Q’s are numeric
print("Q cols dtypes:", df[["Q1","Q2","Q3","Q4","Q5"]].dtypes, "\n")

# 3) List the unique categories & counts in LLM and Experience
print("LLM distribution:\n", df["LLM"].value_counts(dropna=False), "\n")
print("Experience distribution:\n", df["Experience"].value_counts(dropna=False), "\n")

# 4) Finally print the computed Avg summary again
df["Avg"] = df[["Q1","Q2","Q3","Q4","Q5"]].mean(axis=1)
print("Avg summary:\n", df["Avg"].describe())
