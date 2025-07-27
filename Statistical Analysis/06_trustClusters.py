from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and clean data
df = pd.read_excel("AnalysisForGPT.xlsx")
df = df.iloc[1:]  # Skip duplicate header
df.columns = [
    "LLM", "Proficiency", "UsedAITool", "Gen1", "Gen2", "Gen3",
    "Gen4", "Gen5", "Gen6"
]
trust_questions = ["Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6"]
df[trust_questions] = df[trust_questions].apply(pd.to_numeric, errors="coerce")
trust_data = df[trust_questions].dropna()

# Step 1: Standardize the trust question responses
scaler = StandardScaler()
X_scaled = scaler.fit_transform(trust_data)

# Step 2: Run KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["TrustCluster"] = kmeans.fit_predict(X_scaled)

# Step 3: Get the cluster centroids (in original scale)
centroids = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=trust_questions
)
centroids["Cluster"] = centroids.index

# Show rounded centroids
print(centroids.round(2))

import matplotlib.pyplot as plt
import seaborn as sns

# Plot cluster profiles using a grouped bar chart
plt.figure(figsize=(10, 6))
centroids.set_index("Cluster").T.plot(kind="bar", figsize=(12, 6), colormap="Set2")

# Add labels and title
plt.title("Trust Behavior Profiles by Cluster")
plt.ylabel("Average Rating (1–5)")
plt.xticks(rotation=0)
plt.legend(title="Cluster", labels=["Trust Enthusiasts", "Skeptics", "Balanced Evaluators"])
plt.tight_layout()

# Show the plot
plt.show()

