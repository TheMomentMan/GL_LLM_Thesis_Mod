import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the dataset
df = pd.read_excel("All_LLM_Data_everything.xlsx", sheet_name="Sheet1")

# Step 2: Select Gen3 to Gen8 columns and drop missing
gen_cols = ["Gen3", "Gen4", "Gen5", "Gen6", "Gen7", "Gen8"]
gen_data = df[gen_cols].dropna()

# Step 3: Standardize the data
scaler = StandardScaler()
gen_scaled = scaler.fit_transform(gen_data)

# Step 4: Run PCA
pca = PCA()
pca.fit(gen_scaled)

# Step 5: Scree Plot
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
plt.title('Scree Plot for Gen3–Gen8')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, len(explained_variance) + 1))
plt.tight_layout()
plt.show()

# Step 6: Compute Loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings_df = pd.DataFrame(loadings, index=gen_cols, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

# Display loadings
print("\nPCA Loadings (Gen3–Gen8):")
print(loadings_df.round(3))
