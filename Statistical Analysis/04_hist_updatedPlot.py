import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import numpy as np

# Load your data
df = pd.read_excel("AnalysisForGPT.xlsx")

# Verify column names and then set these exactly:
questions = ["Gen3", "Gen4", "Gen5", "Gen6", "Gen7", "Gen8"]
influencers=["Clarity","Confidence Score","Sources Citation", "Friendly Tone","Technical language","Hedging"]

# 2) Make axes
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

# 3) Grab a “Blues” ramp from 20% up (to avoid the near-white)
cmap = mpl.cm.get_cmap("Blues")
blue_colors = [cmap(x) for x in np.linspace(0.2, 1, 5)]

for ax, q,r  in zip(axes, questions, influencers):
    # countplot with our five chosen blues
    sns.countplot(
        x=q,
        data=df,
        order=[1,2,3,4,5],
        palette=blue_colors,
        ax=ax
    )
    # horizontal grid only
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.set_title(f"{r} Response Distribution", fontsize=14)
    ax.set_xlabel("Rating (1–5)")
    ax.set_ylabel("Count")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["1","2","3","4","5"])

    ax.set_ylim(0, 70)  # extend y-axis for clarity
    ax.set_yticklabels([])

    # annotate
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(
            f"{h:.0f}",
            (p.get_x() + p.get_width()/2, h + 0.3),
            ha='center', va='bottom', fontsize=12
        )

# hide unused
for ax in axes[len(questions):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
