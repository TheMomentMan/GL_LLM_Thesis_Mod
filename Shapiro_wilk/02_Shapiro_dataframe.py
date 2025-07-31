from scipy.stats import shapiro
import pandas as pd

# Load the dataset
df = pd.read_csv('Experience_LLM_30thJuly_1349.csv')

# Calculate average trust score
df['TrustScore'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean(axis=1)

# Initialize a list to store results
results = []

# Loop through each LLM group
for group in df['LLM'].unique():
    data = df[df['LLM'] == group]['TrustScore']
    stat, p = shapiro(data)
    normality = "Normal" if p > 0.05 else "Non-normal"
    results.append([group, round(stat, 3), round(p, 3), normality])

# Convert results to DataFrame
result_df = pd.DataFrame(results, columns=['LLM', 'W-statistic', 'p-value', 'Normality'])
result_df.to_csv('shapiro_results.csv', index=False)

# Display result as a 4x4 table
print(result_df)
