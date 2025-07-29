import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV with encoding fix
df = pd.read_csv("LLM_Comm_trust_Resp.csv", encoding="ISO-8859-1")

# Add a column to identify respondent (optional if not already indexed)
df["Respondent"] = df.index

# Melt the LLM responses from wide to long
llm_cols = [f"Q{i}_LLM" for i in range(1, 6)]
df_long = df.melt(id_vars=["LLM", "Respondent"], value_vars=llm_cols,
                  var_name="Question", value_name="LLM_Response")

# Compute sentiment score for each response
df_long["Sentiment"] = df_long["LLM_Response"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Average sentiment score per LLM
sentiment_summary = df_long.groupby("LLM")["Sentiment"].agg(["mean", "std", "count"]).round(3).reset_index()
sentiment_summary.columns = ["LLM", "Avg_Sentiment", "Std_Dev", "Num_Responses"]

print(sentiment_summary)

# Optional: visualize
sns.barplot(data=sentiment_summary, x="LLM", y="Avg_Sentiment", palette="coolwarm")
plt.title("Average Sentiment Score per LLM Persona")
plt.ylabel("Average Sentiment (TextBlob)")
plt.xlabel("LLM Persona")
plt.ylim(-1, 1)
plt.axhline(0, linestyle="--", color="gray", alpha=0.7)
plt.tight_layout()
plt.show()
