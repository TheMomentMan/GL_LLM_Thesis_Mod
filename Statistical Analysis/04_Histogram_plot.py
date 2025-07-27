import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel("AnalysisForGPT.xlsx")
df = df.iloc[1:]  # Skip duplicate header
df.columns = [
    "LLM", "Proficiency", "UsedAITool", "Clarity", "Confidence Scores", "Sources Citation", "Friendly Tone", 
    "Technical Language", "Hedging"
]
trust_questions = ["Clarity", "Confidence Scores", "Sources Citation", "Friendly Tone", 
    "Technical Language", "Hedging"]
df[trust_questions] = df[trust_questions].apply(pd.to_numeric, errors="coerce")

# Calculate distribution
#distribution = {q: df[q].value_counts().sort_index() for q in trust_questions}
distribution = {
    q: df[q].value_counts().sort_index().reindex([1, 2, 3, 4, 5], fill_value=0)
    for q in trust_questions
}

# Plot: grid of response distributions
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for idx, question in enumerate(trust_questions):
    sns.barplot(
        #x=distribution[question].index,
        x=[str(i) for i in distribution[question].index],
        y=distribution[question].values,
        ax=axes[idx],
        palette="Blues_d"
    )
    axes[idx].set_title(f"{question} Response Distribution")
    axes[idx].set_xlabel("Rating (1–5)")
    axes[idx].set_ylabel("Count")
    #axes[idx].set_xticks([1, 2, 3, 4, 5])
    axes[idx].set_xticks([1, 2, 3, 4, 5])
    axes[idx].set_xticklabels(['1', '2', '3', '4', '5'])
    axes[idx].tick_params(axis='x', labelrotation=0)


plt.tight_layout()
plt.show()
