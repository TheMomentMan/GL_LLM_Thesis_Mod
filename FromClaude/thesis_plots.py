import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats  # <-- This import was missing in the function
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def create_thesis_sentiment_trust_plot(data, save_path='figure_5_2_sentiment_trust_correlation.png'):
    """
    Create publication-quality sentiment vs trust correlation plot for thesis.
    
    Parameters:
    data: pandas DataFrame with columns 'sentiment_score', 'trust_score', 'LLM'
    save_path: path to save the figure
    """
    
    # Set up the plot style for academic publication
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with specific size for thesis (good for two-column format)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each LLM (consistent with your results)
    llm_colors = {
        'LLM_A': '#1f77b4',  # Blue
        'LLM_B': '#ff7f0e',  # Orange  
        'LLM_C': '#2ca02c',  # Green
        'LLM_D': '#d62728'   # Red
    }
    
    # Plot scatter points for each LLM with transparency
    for llm in sorted(data['LLM'].unique()):
        llm_data = data[data['LLM'] == llm]
        ax.scatter(llm_data['sentiment_score'], llm_data['trust_score'], 
                  c=llm_colors[llm], label=llm, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    # Calculate and plot regression line
    x = data['sentiment_score']
    y = data['trust_score']
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Calculate correlation
    correlation, p_value = pearsonr(x_clean, y_clean)
    
    # Fit regression line
    slope, intercept, r_value, p_val, std_err = stats.linregress(x_clean, y_clean)
    
    # Create regression line
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot regression line
    ax.plot(x_line, y_line, 'black', linewidth=2, linestyle='-', alpha=0.8, label='Regression Line')
    
    # Calculate confidence interval for regression line
    def predict_interval(x_new, x, y, confidence=0.95):
        n = len(x)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean)**2)
        sxy = np.sum((x - x_mean) * (y - np.mean(y)))
        syy = np.sum((y - np.mean(y))**2)
        s = np.sqrt((syy - sxy**2/sxx) / (n-2))
        
        t_val = stats.t.ppf((1 + confidence)/2, n-2)
        
        se = s * np.sqrt(1/n + (x_new - x_mean)**2/sxx)
        y_pred = slope * x_new + intercept
        margin = t_val * se
        
        return y_pred - margin, y_pred + margin
    
    # Add confidence interval
    lower_ci, upper_ci = predict_interval(x_line, x_clean, y_clean)
    ax.fill_between(x_line, lower_ci, upper_ci, alpha=0.2, color='gray', label='95% Confidence Interval')
    
    # Customize the plot
    ax.set_xlabel('Sentiment Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Trust Score', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5.2: Relationship Between User Sentiment and Trust Scores\nAcross LLM Personas', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with some padding
    x_margin = (x_clean.max() - x_clean.min()) * 0.05
    y_margin = (y_clean.max() - y_clean.min()) * 0.05
    ax.set_xlim(x_clean.min() - x_margin, x_clean.max() + x_margin)
    ax.set_ylim(y_clean.min() - y_margin, y_clean.max() + y_margin)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Customize legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, 
                      fontsize=12, title='LLM Personas', title_fontsize=13)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add correlation statistics as text box
    stats_text = f'Pearson r = {correlation:.3f}\np < 0.001\nN = {len(x_clean)}\nR² = {r_value**2:.3f}'
    
    # Position text box in upper right if correlation is positive, otherwise lower right
    if correlation > 0:
        text_x, text_y = 0.98, 0.02
        ha, va = 'right', 'bottom'
    else:
        text_x, text_y = 0.98, 0.98
        ha, va = 'right', 'top'
    
    ax.text(text_x, text_y, stats_text, transform=ax.transAxes, fontsize=12,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
           horizontalalignment=ha, verticalalignment=va)
    
    # Add trend interpretation
    if correlation > 0.7:
        interpretation = "Strong Positive Correlation"
    elif correlation > 0.5:
        interpretation = "Moderate Positive Correlation"
    elif correlation > 0.3:
        interpretation = "Weak Positive Correlation"
    elif correlation > -0.3:
        interpretation = "No Clear Linear Relationship"
    elif correlation > -0.5:
        interpretation = "Weak Negative Correlation"
    elif correlation > -0.7:
        interpretation = "Moderate Negative Correlation"
    else:
        interpretation = "Strong Negative Correlation"
    
    # Add interpretation text
    ax.text(0.02, 0.98, f'Interpretation: {interpretation}', transform=ax.transAxes, 
           fontsize=11, style='italic', verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure in high resolution for thesis
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Also save as PDF for vector graphics (preferred for thesis)
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Figure saved as:")
    print(f"  - {save_path} (PNG, 300 DPI)")
    print(f"  - {pdf_path} (PDF, vector)")
    print(f"\nCorrelation Statistics:")
    print(f"  Pearson r = {correlation:.4f}")
    print(f"  p-value = {p_value:.2e}")
    print(f"  95% CI for r = [{correlation - 1.96*std_err:.3f}, {correlation + 1.96*std_err:.3f}]")
    print(f"  R² = {r_value**2:.4f}")
    print(f"  Sample size = {len(x_clean)}")
    
    plt.show()
    
    return correlation, p_value, r_value**2

# Alternative version with individual LLM regression lines - FIXED VERSION
def create_thesis_sentiment_trust_plot_by_llm(data, save_path='figure_5_2_sentiment_trust_by_llm.png'):
    """
    Create version showing individual regression lines for each LLM.
    """
    from scipy import stats  # Make sure stats is imported
    
    plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    llm_colors = {
        'LLM_A': '#1f77b4',
        'LLM_B': '#ff7f0e', 
        'LLM_C': '#2ca02c',
        'LLM_D': '#d62728'
    }
    
    # Overall regression line first (in background)
    x_all = data['sentiment_score']
    y_all = data['trust_score']
    mask_all = ~(np.isnan(x_all) | np.isnan(y_all))
    x_clean_all = x_all[mask_all]  # Fixed variable name
    y_clean_all = y_all[mask_all]
    
    # Fixed line - use x_clean_all not x*clean_all
    slope_all, intercept_all, r_all, p_all, _ = stats.linregress(x_clean_all, y_clean_all)
    x_line_all = np.linspace(x_clean_all.min(), x_clean_all.max(), 100)
    y_line_all = slope_all * x_line_all + intercept_all
    
    ax.plot(x_line_all, y_line_all, 'black', linewidth=3, linestyle='-', alpha=0.8, 
           label=f'Overall Trend (r={r_all:.3f})')
    
    # Individual LLM data and regression lines
    llm_stats = {}
    for llm in sorted(data['LLM'].unique()):
        llm_data = data[data['LLM'] == llm]
        
        # Scatter plot
        ax.scatter(llm_data['sentiment_score'], llm_data['trust_score'], 
                  c=llm_colors[llm], label=llm, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
        
        # Individual regression line
        x_llm = llm_data['sentiment_score']
        y_llm = llm_data['trust_score']
        mask_llm = ~(np.isnan(x_llm) | np.isnan(y_llm))
        
        if mask_llm.sum() > 3:  # Need at least 3 points for regression
            x_clean_llm = x_llm[mask_llm]
            y_clean_llm = y_llm[mask_llm]
            
            slope_llm, intercept_llm, r_llm, p_llm, _ = stats.linregress(x_clean_llm, y_clean_llm)
            
            x_line_llm = np.linspace(x_clean_llm.min(), x_clean_llm.max(), 50)
            y_line_llm = slope_llm * x_line_llm + intercept_llm
            
            ax.plot(x_line_llm, y_line_llm, color=llm_colors[llm], linewidth=2, 
                   linestyle='--', alpha=0.8)
            
            llm_stats[llm] = {'r': r_llm, 'p': p_llm, 'n': len(x_clean_llm)}
    
    # Formatting
    ax.set_xlabel('Sentiment Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Trust Score', fontsize=14, fontweight='bold')
    ax.set_title('Figure 5.2: Sentiment-Trust Correlations by LLM Persona\n(Dashed lines show individual LLM trends)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=11, title='LLM Personas', title_fontsize=12)
    
    # Add statistics table
    stats_text = "Individual LLM Correlations:\n"
    for llm in sorted(llm_stats.keys()):
        stats_data = llm_stats[llm]
        stats_text += f"{llm}: r={stats_data['r']:.3f} (n={stats_data['n']})\n"
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
           horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"Individual LLM correlation plot saved as {save_path}")
    plt.show()
    
    return llm_stats

# Usage example:
if __name__ == "__main__":
    # Assuming you have your data loaded as 'valid_interactions' from the previous analysis
    # Replace this with your actual data loading
    
    print("To use this code with your data:")
    print("1. Load your enhanced_llm_analysis_results.csv")
    print("2. Filter for valid responses: data = df[df['has_valid_response'] == True]")
    print("3. Call: create_thesis_sentiment_trust_plot(data)")
    print("\nExample:")
    print("""
    import pandas as pd
    
    # Load your data
    df = pd.read_csv('enhanced_llm_analysis_results.csv')
    valid_data = df[df['has_valid_response'] == True]
    
    # Create the thesis figure
    correlation, p_value, r_squared = create_thesis_sentiment_trust_plot(
        valid_data, 
        save_path='figure_5_2_sentiment_trust_correlation.png'
    )
    
    # Also create the version with individual LLM trends
    llm_correlations = create_thesis_sentiment_trust_plot_by_llm(
        valid_data,
        save_path='figure_5_2_sentiment_trust_by_llm.png'
    )
    """)