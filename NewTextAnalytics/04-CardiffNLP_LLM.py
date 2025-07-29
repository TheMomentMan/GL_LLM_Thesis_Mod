import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm

# Load your data
df = pd.read_csv("LLM_Comm_trust_Resp.csv")

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Identify LLM response columns
llm_columns = [col for col in df.columns if col.startswith("Q") and col.endswith("_LLM")]

# Function to compute sentiment
def get_sentiment_score(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    return float(probs[2]) - float(probs[0])  # Compound: pos - neg

# Compute sentiment per respondent
avg_scores = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    scores = [get_sentiment_score(row[col]) for col in llm_columns if pd.notna(row[col])]
    avg_scores.append(sum(scores)/len(scores) if scores else None)
df["Avg_LLM_Sentiment"] = avg_scores

# Save updated DataFrame
df.to_csv("LLM_Comm_trust_Resp_with_sentiment.csv", index=False)

# Summarize by LLM
summary = (
    df.groupby("LLM")["Avg_LLM_Sentiment"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "Avg_Sentiment", "std": "Std_Dev", "count": "Num_Responses"})
)

# Display summary
print("\nSentiment Summary per LLM:\n")
print(summary.round(3))

# Save summary
summary.to_csv("LLM_sentiment_summary2.csv", index=False)
