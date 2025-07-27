import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with the appropriate encoding
df = pd.read_csv("Respondents_Comments1_5.csv", encoding='ISO-8859-1')  # or encoding='latin1'

# Identify comment columns (adjust if named differently)
comment_cols = [col for col in df.columns if col.lower().startswith('comment')]

# Compute sentiment polarity for each comment column
for col in comment_cols:
    df[f'Sentiment_{col[-1]}'] = df[col].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)

# Calculate average sentiment score per respondent
sentiment_cols = [f'Sentiment_{col[-1]}' for col in comment_cols]
df['Avg_Sentiment'] = df[sentiment_cols].mean(axis=1)

# Save updated dataframe with sentiment columns
df.to_csv("Sentiment_Annotated_Comments.csv", index=False)

# Plot 1: Distribution of Average Sentiment Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Avg_Sentiment'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Average Sentiment Scores')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_sentiment_distribution.png")
plt.close()

# Plot 2: Boxplot of sentiment scores per comment
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[sentiment_cols], palette="pastel")
plt.title('Boxplot of Sentiment Scores per Comment')
plt.xlabel('Comment')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_sentiments.png")
plt.close()
