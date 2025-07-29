#!/usr/bin/env python3
"""
Complete LLM Communication-Trust Analysis Implementation
========================================================

This script replicates and enhances the analysis done in Claude's JavaScript environment
using professional Python libraries for production-quality results.

Requirements:
pip install pandas numpy scipy scikit-learn matplotlib seaborn nltk textblob wordcloud plotly

Author: Analysis framework developed with Claude
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    from textblob import TextBlob
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    ADVANCED_NLP = True
except ImportError:
    print("Advanced NLP libraries not available. Using basic sentiment analysis.")
    ADVANCED_NLP = False

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Advanced plotting
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class LLMTrustAnalyzer:
    """
    Comprehensive LLM Communication-Trust Analysis Tool
    
    This class provides all the functionality to analyze LLM communication patterns
    and their relationship to user trust scores.
    """
    
    def __init__(self, csv_path):
        """Initialize the analyzer with data from CSV file."""
        self.data = None
        self.processed_data = None
        self.sentiment_analyzer = None
        self.load_data(csv_path)
        self.setup_sentiment_analyzer()
        
    def load_data(self, csv_path):
        """Load and initial processing of the CSV data."""
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_path, encoding='cp1252')
        
        print(f"Loaded {len(self.data)} participants")
        print(f"Columns: {list(self.data.columns)}")
        
    def setup_sentiment_analyzer(self):
        """Setup sentiment analysis tools."""
        if ADVANCED_NLP:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """
        Comprehensive sentiment analysis using multiple approaches.
        
        Returns:
            dict: Sentiment scores from different methods
        """
        if not isinstance(text, str) or text.strip() == '' or text == 'undefined':
            return {
                'compound': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 1,
                'textblob_polarity': 0,
                'textblob_subjectivity': 0,
                'custom_score': 0,
                'label': 'neutral',
                'confidence': 0
            }
        
        results = {}
        
        # VADER sentiment (if available)
        if ADVANCED_NLP and self.sentiment_analyzer:
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            results.update(vader_scores)
        
        # TextBlob sentiment (if available)
        if ADVANCED_NLP:
            blob = TextBlob(text)
            results['textblob_polarity'] = blob.sentiment.polarity
            results['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # Custom trust-specific sentiment analysis
        custom_score = self._custom_sentiment_analysis(text)
        results.update(custom_score)
        
        return results
    
    def _custom_sentiment_analysis(self, text):
    #Enhanced sentiment analysis matching JavaScript version
        if not text or text == 'undefined' or len(str(text).strip()) < 3:
            return {'custom_score': 0, 'label': 'neutral', 'confidence': 0}
        
        text_lower = str(text).lower()
        
        # More comprehensive lexicons
        trust_builders = [
            'makes sense', 'good point', 'helpful', 'relevant', 'appropriate', 
            'thorough', 'detailed', 'useful', 'insightful', 'reasonable',
            'logical', 'sound', 'smart', 'good', 'right', 'correct', 'clear',
            'professional', 'comprehensive', 'valuable', 'effective'
        ]
        
        trust_killers = [
            'generic', 'basic', 'not relevant', 'unclear', 'confusing',
            'not helpful', 'wrong', 'bad', 'useless', 'pointless',
            'too simple', 'obvious', 'not applicable', 'doesn\'t apply'
        ]
        
        positive_count = sum(1 for phrase in trust_builders if phrase in text_lower)
        negative_count = sum(1 for phrase in trust_killers if phrase in text_lower)
        
        # More sensitive scoring
        score = (positive_count * 1.5) - (negative_count * 2.0)
        word_count = len(text_lower.split())
        normalized_score = score / max(np.sqrt(word_count), 1)
        
        # More sensitive thresholds
        if normalized_score > 0.2: label = 'positive'
        elif normalized_score < -0.2: label = 'negative'  
        else: label = 'neutral'
        
        return {
            'custom_score': normalized_score,
            'label': label,
            'confidence': min(abs(normalized_score), 1.0)
    }
    
    def analyze_llm_communication_style(self, text):
        """Analyze LLM communication characteristics."""
        if not isinstance(text, str) or text.strip() == '':
            return self._empty_style_analysis()
        
        # Basic metrics
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Communication style features
        is_question = text.strip().endswith('?')
        question_count = text.count('?')
        has_numbers = bool(re.search(r'\d', text))
        
        # Pattern matching
        patterns = {
            'starts_with_question': bool(re.match(r'^(would|have|could|should|can|do|did|will|what|how|why|when|where)\b', text, re.I)),
            'has_technical_terms': bool(re.search(r'\b(pressure|flow|temperature|valve|pump|injection|production|reservoir|wellhead|tubing|gas|oil|water|GOR|survey|analysis|model|acoustic|membrane|separator|choke)\b', text, re.I)),
            'has_uncertainty': bool(re.search(r'\b(might|could|maybe|perhaps|possibly|probably|likely|uncertain|unsure|may)\b', text, re.I)),
            'has_confidence': bool(re.search(r'\b(definitely|certainly|clearly|obviously|sure|confirm|establish|determine|will|should|must)\b', text, re.I)),
            'is_directive': bool(re.search(r'\b(should|must|need to|have to|require|recommend|suggest)\b', text, re.I)),
            'is_collaborative': bool(re.search(r'\b(we|us|our|together|work with|help|assist)\b', text, re.I)),
            'is_explorative': bool(re.search(r'\b(explore|investigate|check|examine|look into|consider|review)\b', text, re.I))
        }
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
            'is_question': is_question,
            'question_count': question_count,
            'has_numbers': has_numbers,
            **patterns
        }
    
    def _empty_style_analysis(self):
        """Return empty analysis for missing text."""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_words_per_sentence': 0, 'is_question': False, 'question_count': 0,
            'has_numbers': False, 'starts_with_question': False, 'has_technical_terms': False,
            'has_uncertainty': False, 'has_confidence': False, 'is_directive': False,
            'is_collaborative': False, 'is_explorative': False
        }
    
    def process_all_data(self):
        """Process all interactions and create comprehensive dataset."""
        processed_rows = []
        
        for idx, row in self.data.iterrows():
            for q in range(1, 6):  # Q1 through Q5
                user_response = row.get(f'Q{q}resp', '')
                trust_score = row.get(f'Q{q}_Trust', row.get(f'Q{q}_trust', None))
                llm_text = row.get(f'Q{q}_LLM', '')
                
                if pd.notna(trust_score):
                    # Analyze sentiment of user response
                    sentiment = self.analyze_sentiment(user_response)
                    
                    # Analyze LLM communication style
                    llm_style = self.analyze_llm_communication_style(llm_text)
                    
                    processed_row = {
                        'participant_id': idx + 1,
                        'LLM': row['LLM'],
                        'Experience': row['Experience'],
                        'question': q,
                        'trust_score': trust_score,
                        'user_response': user_response,
                        'llm_text': llm_text,
                        'has_valid_response': len(str(user_response).strip()) > 0 and str(user_response) != 'undefined',
                        **{f'sentiment_{k}': v for k, v in sentiment.items()},
                        **{f'llm_{k}': v for k, v in llm_style.items()}
                    }
                    processed_rows.append(processed_row)
        
        self.processed_data = pd.DataFrame(processed_rows)
        print(f"Processed {len(self.processed_data)} interactions")
        print(f"Valid responses: {self.processed_data['has_valid_response'].sum()}")
        
        return self.processed_data
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        if self.processed_data is None:
            self.process_all_data()
        
        # Filter valid responses
        valid_data = self.processed_data[self.processed_data['has_valid_response']]
        
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Overall statistics
        print(f"Total interactions: {len(self.processed_data)}")
        print(f"Valid responses: {len(valid_data)} ({len(valid_data)/len(self.processed_data)*100:.1f}%)")
        
        # Sentiment distribution
        print(f"\nSentiment Distribution:")
        sentiment_dist = valid_data['sentiment_label'].value_counts()
        for label, count in sentiment_dist.items():
            print(f"  {label}: {count} ({count/len(valid_data)*100:.1f}%)")
        
        # LLM performance
        print(f"\nLLM Performance Summary:")
        llm_stats = valid_data.groupby('LLM').agg({
            'trust_score': ['mean', 'std', 'count'],
            'sentiment_custom_score': ['mean', 'std'],
            'llm_char_count': 'mean',
            'llm_question_count': 'mean',
            'llm_has_technical_terms': 'mean'
        }).round(3)
        
        print(llm_stats)
        
        # Correlations
        print(f"\nKey Correlations:")
        correlations = {
            'Sentiment-Trust': pearsonr(valid_data['sentiment_custom_score'], valid_data['trust_score'])[0],
            'Length-Trust': pearsonr(valid_data['llm_char_count'], valid_data['trust_score'])[0],
            'Questions-Trust': pearsonr(valid_data['llm_question_count'], valid_data['trust_score'])[0],
            'Technical-Trust': pearsonr(valid_data['llm_has_technical_terms'].astype(int), valid_data['trust_score'])[0]
        }
        
        for name, corr in correlations.items():
            print(f"  {name}: r = {corr:.3f}")
        
        return llm_stats, correlations
    
    

    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations."""
        if self.processed_data is None:
            self.process_all_data()
        
        valid_data = self.processed_data[self.processed_data['has_valid_response']]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LLM Communication-Trust Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Trust distribution by LLM
        trust_by_llm = valid_data.groupby(['LLM', 'trust_score']).size().unstack(fill_value=0)
        trust_by_llm.plot(kind='bar', ax=axes[0,0], stacked=True)
        axes[0,0].set_title('Trust Score Distribution by LLM')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend(title='Trust Score')
        
        # 2. Sentiment vs Trust scatter
        for llm in valid_data['LLM'].unique():
            llm_data = valid_data[valid_data['LLM'] == llm]
            axes[0,1].scatter(llm_data['sentiment_custom_score'], llm_data['trust_score'], 
                            label=llm, alpha=0.6)
        axes[0,1].set_title('Sentiment vs Trust Score')
        axes[0,1].set_xlabel('Sentiment Score')
        axes[0,1].set_ylabel('Trust Score')
        axes[0,1].legend()
        
        # 3. LLM response length distribution
        valid_data.boxplot(column='llm_char_count', by='LLM', ax=axes[0,2])
        axes[0,2].set_title('Response Length by LLM')
        axes[0,2].set_ylabel('Character Count')
        
        # 4. Average trust by LLM
        trust_means = valid_data.groupby('LLM')['trust_score'].mean()
        trust_means.plot(kind='bar', ax=axes[1,0], color='skyblue')
        axes[1,0].set_title('Average Trust Score by LLM')
        axes[1,0].set_ylabel('Average Trust Score')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 5. Sentiment distribution
        valid_data['sentiment_label'].value_counts().plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
        axes[1,1].set_title('Overall Sentiment Distribution')
        
        # 6. Communication style heatmap
        style_features = ['llm_is_question', 'llm_has_technical_terms', 'llm_has_uncertainty', 
                         'llm_has_confidence', 'llm_is_directive', 'llm_is_collaborative']
        style_by_llm = valid_data.groupby('LLM')[style_features].mean()
        sns.heatmap(style_by_llm.T, annot=True, fmt='.2f', ax=axes[1,2], cmap='YlOrRd')
        axes[1,2].set_title('Communication Style by LLM')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('llm_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            print("Dashboard saved as 'llm_analysis_dashboard.png'")
        
        plt.show()
        
        # Create interactive plots if Plotly is available
        if PLOTLY_AVAILABLE:
            self._create_interactive_plots(valid_data)
    
    def _create_interactive_plots(self, data):
        """Create interactive Plotly visualizations."""
        # Interactive sentiment-trust relationship
        fig = px.scatter(data, x='sentiment_custom_score', y='trust_score', 
                        color='LLM', size='llm_char_count',
                        hover_data=['Experience', 'llm_question_count'],
                        title='Interactive Sentiment vs Trust Analysis')
        fig.write_html('sentiment_trust_interactive.html')
        print("Interactive plot saved as 'sentiment_trust_interactive.html'")
    
    def build_trust_prediction_model(self):
        """Build machine learning model to predict trust from communication features."""
        if self.processed_data is None:
            self.process_all_data()
        
        valid_data = self.processed_data[self.processed_data['has_valid_response']].copy()
        
        # Feature engineering
        feature_columns = [
            'llm_char_count', 'llm_word_count', 'llm_question_count',
            'llm_has_technical_terms', 'llm_has_uncertainty', 'llm_has_confidence',
            'llm_is_directive', 'llm_is_collaborative', 'llm_is_explorative',
            'sentiment_custom_score'
        ]
        
        # Convert boolean columns to int
        for col in feature_columns:
            if valid_data[col].dtype == bool:
                valid_data[col] = valid_data[col].astype(int)
        
        X = valid_data[feature_columns]
        y = valid_data['trust_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"{name}:")
            print(f"  R² Score: {r2:.3f}")
            print(f"  RMSE: {np.sqrt(mse):.3f}")
            
            # Feature importance for Random Forest
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("  Top 5 Important Features:")
                for _, row in importance_df.head().iterrows():
                    print(f"    {row['feature']}: {row['importance']:.3f}")
            print()
        
        return results, X_test, y_test
    
    def export_results(self, filename='llm_analysis_results.csv'):
        """Export processed data and results to CSV."""
        if self.processed_data is None:
            self.process_all_data()
        
        self.processed_data.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = LLMTrustAnalyzer('LLM_Comm_trust_Resp.csv')
    
    # Process data
    processed_data = analyzer.process_all_data()
    
    # Generate summary statistics
    llm_stats, correlations = analyzer.generate_summary_statistics()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Build prediction model
    model_results, X_test, y_test = analyzer.build_trust_prediction_model()
    
    # Export results
    analyzer.export_results()
    
    print("\nAnalysis complete! Check the generated files:")
    print("- llm_analysis_dashboard.png: Visual dashboard")
    print("- llm_analysis_results.csv: Complete processed dataset")
    print("- sentiment_trust_interactive.html: Interactive plots (if Plotly available)")

if __name__ == "__main__":
    main()