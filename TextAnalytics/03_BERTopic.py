from bertopic import BERTopic
import pandas as pd

# Load your preprocessed data
df = pd.read_csv("Alltextresponses_preprocessed.csv")

# Drop rows where cleaned_text is missing
df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)

# Fit BERTopic on the cleaned text column
topic_model = BERTopic(language="english", verbose=True)
topics, _ = topic_model.fit_transform(df["cleaned_text"].tolist())

# Show basic topic info
print(topic_model.get_topic_info())

# Optionally: print top words for each topic
for topic_num in topic_model.get_topic_info().Topic[1:]:  # skip outlier topic -1
    print(f"\nTopic {topic_num}:")
    for word, weight in topic_model.get_topic(topic_num):
        print(f"  {word} ({round(weight, 4)})")
