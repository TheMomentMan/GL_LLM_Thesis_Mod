# run_thesis_plot.py
import pandas as pd
from thesis_plots import create_thesis_sentiment_trust_plot, create_thesis_sentiment_trust_plot_by_llm

# Load your data
print("Loading data...")
df = pd.read_csv('enhanced_llm_analysis_results.csv')

# Filter for valid responses only
valid_data = df[df['has_valid_response'] == True]
print(f"Valid interactions: {len(valid_data)}")

# Create the main thesis figure
print("\nCreating Figure 5.2 (main version)...")
correlation, p_value, r_squared = create_thesis_sentiment_trust_plot(
    valid_data, 
    save_path='figure_5_2_sentiment_trust_correlation.png'
)

# Create the version with individual LLM regression lines
print("\nCreating Figure 5.2 (LLM-specific version)...")
llm_correlations = create_thesis_sentiment_trust_plot_by_llm(
    valid_data,
    save_path='figure_5_2_sentiment_trust_by_llm.png'
)

print("\n✅ All figures created successfully!")
print("Check your folder for:")
print("- figure_5_2_sentiment_trust_correlation.png")
print("- figure_5_2_sentiment_trust_correlation.pdf") 
print("- figure_5_2_sentiment_trust_by_llm.png")
print("- figure_5_2_sentiment_trust_by_llm.pdf")