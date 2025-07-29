import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load the file with the correct encoding
df = pd.read_csv("LLM_Comm_trust_Resp.csv", encoding="ISO-8859-1")

# Prepare column lists
llm_cols = [f"Q{i}_LLM" for i in range(1, 6)]
trust_cols = [f"Q{i}_Trust" for i in range(1, 6)]

# Melt both into long format
llm_melt = df[llm_cols].melt(var_name="Question", value_name="LLM_Response")
trust_melt = df[trust_cols].melt(var_name="Question", value_name="Trust_Score")

# Normalize question labels
llm_melt["Question"] = llm_melt["Question"].str.extract(r"(Q\d)")
trust_melt["Question"] = trust_melt["Question"].str.extract(r"(Q\d)")

# Combine LLM responses and Trust scores
combined = pd.concat([llm_melt["Question"], llm_melt["LLM_Response"], trust_melt["Trust_Score"]], axis=1)

# Sentiment score using TextBlob
combined["Sentiment"] = combined["LLM_Response"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Show average sentiment per trust score
avg_sentiment = combined.groupby("Trust_Score")["Sentiment"].mean().reset_index()
print(combined.head())

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.boxplot(data=combined, x="Trust_Score", y="Sentiment", palette="coolwarm")
plt.title("LLM Response Sentiment vs Trust Score")
plt.xlabel("Trust Score (1–5)")
plt.ylabel("Sentiment Score")
plt.tight_layout()
plt.show()
