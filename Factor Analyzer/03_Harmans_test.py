import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load your dataset
df = pd.read_excel("All_LLM_Data_everything.xlsx")  # Replace with actual file path

# Step 2: Select the trust rating columns (Likert-scale only)
cmb_cols = ["Q1", "Q4", "Q7", "Q10", "Q13"]
cmb_data = df[cmb_cols].apply(pd.to_numeric, errors='coerce').dropna()

# Step 3: Standardize the data
scaler = StandardScaler()
cmb_scaled = scaler.fit_transform(cmb_data)

# Step 4: Run PCA
pca = PCA()
pca.fit(cmb_scaled)

# Step 5: Check variance explained
explained_var = pca.explained_variance_ratio_
print("Variance explained by first component:", explained_var[0])
print("All components:", explained_var.round(3))
