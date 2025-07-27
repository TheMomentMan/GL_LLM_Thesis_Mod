import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. Load your preprocessed CSV
df = pd.read_csv("Alltextresponses_preprocessed.csv")

# 2. Clean any missing or empty rows
df = df.dropna()
docs = df['cleaned_text'].astype(str).tolist()  # column should match your actual column name

# 3. Convert to document-term matrix using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(docs)
feature_names = vectorizer.get_feature_names_out()

# 4. Fit LDA model
num_topics = 10  # you can change this
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(dtm)

# 5. Print Top 10 Words per Topic
print("\nTop 10 words per topic:\n" + "-"*30)
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx+1}: {', '.join(top_words)}")
