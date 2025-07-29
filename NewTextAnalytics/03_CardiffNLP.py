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

# Columns containing LLM responses
llm_columns = [col for col in df.columns if col.startswith("Q") and col.endswith("_LLM")]

# Function to calculate sentiment
def get_sentiment_score(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    return float(probs[2]) - float(probs[0])  # Compound score

# Compute sentiment per respondent
sentiment_scores = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    row_scores = [get_sentiment_score(row[col]) for col in llm_columns if pd.notna(row[col])]
    sentiment_scores.append(sum(row_scores) / len(row_scores) if row_scores else None)

df['Avg_LLM_Sentiment'] = sentiment_scores

# Save
df.to_csv("LLM_Comm_trust_Resp_with_sentiment.csv", index=False)
