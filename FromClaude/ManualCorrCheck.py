import pandas as pd
import numpy as np

df = pd.read_csv('enhanced_llm_analysis_results.csv')
valid_data = df[df['has_valid_response'] == True]

print("DETAILED SENTIMENT DIAGNOSTIC")
print("="*50)

# 1. Check sentiment score distribution
print(f"\n1. SENTIMENT SCORE DISTRIBUTION (N={len(valid_data)}):")
print(f"   Min: {valid_data['sentiment_score'].min():.3f}")
print(f"   Max: {valid_data['sentiment_score'].max():.3f}")
print(f"   Mean: {valid_data['sentiment_score'].mean():.3f}")
print(f"   Median: {valid_data['sentiment_score'].median():.3f}")
print(f"   Std: {valid_data['sentiment_score'].std():.3f}")

# 2. Count sentiment labels
print(f"\n2. SENTIMENT LABEL COUNTS:")
label_counts = valid_data['sentiment_label'].value_counts()
for label, count in label_counts.items():
    pct = count/len(valid_data)*100
    print(f"   {label}: {count} ({pct:.1f}%)")

# 3. Check trust score distribution for comparison
print(f"\n3. TRUST SCORE DISTRIBUTION:")
trust_counts = valid_data['trust_score'].value_counts().sort_index()
for score, count in trust_counts.items():
    pct = count/len(valid_data)*100
    print(f"   Trust {score}: {count} ({pct:.1f}%)")

# 4. Mean sentiment by trust score
print(f"\n4. MEAN SENTIMENT BY TRUST SCORE:")
for trust in sorted(valid_data['trust_score'].unique()):
    trust_subset = valid_data[valid_data['trust_score'] == trust]
    mean_sent = trust_subset['sentiment_score'].mean()
    print(f"   Trust {trust}: {mean_sent:.3f} (n={len(trust_subset)})")

# 5. Check for sentiment range issues
print(f"\n5. SENTIMENT SCORE RANGES:")
zero_sentiment = (valid_data['sentiment_score'] == 0).sum()
tiny_sentiment = (abs(valid_data['sentiment_score']) < 0.1).sum()
print(f"   Exactly zero: {zero_sentiment} ({zero_sentiment/len(valid_data)*100:.1f}%)")
print(f"   Near zero (±0.1): {tiny_sentiment} ({tiny_sentiment/len(valid_data)*100:.1f}%)")

# 6. Sample actual responses with their sentiment scores
print(f"\n6. SAMPLE RESPONSES AND SENTIMENT SCORES:")
print("   High sentiment examples:")
high_sent = valid_data.nlargest(3, 'sentiment_score')[['user_response', 'sentiment_score', 'trust_score']]
for i, row in high_sent.iterrows():
    print(f"     Score: {row['sentiment_score']:.3f}, Trust: {row['trust_score']}")
    print(f"     Text: '{row['user_response'][:80]}...'")
    print()

print("   Low sentiment examples:")
low_sent = valid_data.nsmallest(3, 'sentiment_score')[['user_response', 'sentiment_score', 'trust_score']]
for i, row in low_sent.iterrows():
    print(f"     Score: {row['sentiment_score']:.3f}, Trust: {row['trust_score']}")
    print(f"     Text: '{row['user_response'][:80]}...'")
    print()

# 7. Check phrase detection
print(f"7. PHRASE DETECTION CHECK:")
phrase_detected = valid_data['sentiment_phrases_detected'].sum()
responses_with_phrases = (valid_data['sentiment_phrases_detected'] > 0).sum()
print(f"   Total phrases detected: {phrase_detected}")
print(f"   Responses with phrases: {responses_with_phrases} ({responses_with_phrases/len(valid_data)*100:.1f}%)")