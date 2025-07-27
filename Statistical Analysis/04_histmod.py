import pandas as pd
import matplotlib.pyplot as plt

def plot_trust_distributions(file_path: str):
    """
    Loads trust-question data from an Excel file and plots
    individual bar charts for each question using matplotlib.
    """
    # 1) Load data (explicitly use openpyxl engine for compatibility)
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.iloc[1:].copy()  # Skip duplicate header row
    
    # 2) Assign clear column names
    df.columns = [
        "LLM", "Proficiency", "UsedAITool",
        "Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6"
    ]
    
    # 3) Convert trust-question columns to numeric
    trust_questions = ["Gen1", "Gen2", "Gen3", "Gen4", "Gen5", "Gen6"]
    df[trust_questions] = df[trust_questions].apply(pd.to_numeric, errors="coerce")
    
    # 4) Plot each question in its own figure
    for question in trust_questions:
        counts = df[question].value_counts().sort_index()
        
        plt.figure()
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f"{question} Response Distribution")
        plt.xlabel("Rating (1–5)")
        plt.ylabel("Count")
        plt.xticks(counts.index)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Update the path if needed:
    plot_trust_distributions("AnalysisForGPT.xlsx")
