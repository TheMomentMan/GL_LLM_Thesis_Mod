import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# You may need to download these once:
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

def load_with_encodings(path, encodings=('utf-8','latin1','cp1252')):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Last resort—ignore errors
    return pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

df = load_with_encodings('Alltextresponses.csv')

# 2) Identify the text column (assumed to be the first one)
text_col = df.columns[0]

# 3) Prepare tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()                              # lowercase
    text = re.sub(r'http\S+', '', text)                   # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)                  # remove punctuation/numbers
    tokens = text.split()                                  
    tokens = [t for t in tokens if t not in stop_words]   # remove stop-words
    tokens = [lemmatizer.lemmatize(t) for t in tokens]     # lemmatize
    tokens = [t for t in tokens if len(t) > 1]             # drop single letters
    return ' '.join(tokens)

# 4) Apply
df['cleaned_text'] = df[text_col].apply(preprocess)

# 5) Inspect & Save
print(df[['cleaned_text']].head())
df.to_csv('Alltextresponses_preprocessed.csv', index=False)
