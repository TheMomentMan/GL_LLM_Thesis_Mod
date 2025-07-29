#!/usr/bin/env python3
"""
Enhanced LLM Communication-Trust Analysis Implementation
======================================================

This script replicates Claude's JavaScript analysis with proper data filtering
and trust-optimized sentiment analysis for accurate results.

Requirements:
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly

Author: Enhanced analysis framework
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

# Machine learning
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
    print("Plotly not available. Skipping interactive plots.")

class EnhancedLLMTrustAnalyzer:
    """
    Enhanced LLM Communication-Trust Analysis Tool
    
    Replicates Claude's JavaScript analysis with proper data filtering
    and trust-optimized sentiment analysis.
    """
    
    def __init__(self, csv_path):
        """Initialize the analyzer with data from CSV file."""
        self.data = None
        self.processed_data = None
        self.valid_interactions = None
        self.load_data(csv_path)
        
    def load_data(self, csv_path):
        """Load and initial processing of the CSV data."""
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(csv_path, encoding='cp1252')
            except:
                self.data = pd.read_csv(csv_path, encoding='latin1')
        
        print(f"Loaded {len(self.data)} participants")
        print(f"Columns: {list(self.data.columns)}")
        
    def is_valid_response(self, text):
        """
        Determine if a user response is valid (not empty/blank/undefined).
        
        This matches Claude's JavaScript filtering logic.
        """
        if pd.isna(text):
            return False
        if text == 'undefined' or text == 'nan' or str(text).lower() == 'nan':
            return False
        if isinstance(text, str):
            cleaned = text.strip()
            if len(cleaned) == 0:
                return False
            if len(cleaned) < 2:  # Very short responses likely not meaningful
                return False
            return True
        return False
    
    def analyze_sentiment(self, text):
        """
        Trust-optimized sentiment analysis matching Claude's JavaScript approach.
        
        Uses expanded lexicons specifically designed for AI trust evaluation.
        """
        if not self.is_valid_response(text):
            return {
                'score': 0,
                'label': 'neutral',
                'confidence': 0,
                'wordCount': 0,
                'positiveWords': 0,
                'negativeWords': 0,
                'phrases': []
            }
        
        text_lower = str(text).lower().strip()
        words = text_lower.split()
        
        # Comprehensive trust-building phrases (from Claude's analysis)
        trust_builders = [
            'makes sense', 'good point', 'helpful', 'relevant', 'appropriate', 'thorough',
            'detailed', 'comprehensive', 'spot on', 'exactly right', 'great suggestion',
            'valuable insight', 'professional', 'knowledgeable', 'logical approach',
            'reasonable', 'practical', 'actionable', 'useful', 'insightful', 'sound advice',
            'clear', 'informative', 'effective', 'smart', 'intelligent', 'correct',
            'good', 'great', 'excellent', 'right', 'proper', 'valid', 'solid',
            'agree', 'accept', 'approve', 'like', 'love', 'appreciate', 'value',
            'trust', 'reliable', 'dependable', 'accurate', 'precise', 'true',
            'will investigate', 'will check', 'will try', 'good idea', 'worth exploring',
            'that could work', 'that helps', 'appreciate the guidance', 'thank you'
        ]
        
        # Comprehensive trust-killing phrases  
        trust_killers = [
            'generic', 'too basic', 'not relevant', 'not applicable', 'wrong approach',
            'doesn\'t apply', 'too general', 'not specific', 'unclear', 'confusing',
            'not helpful', 'unhelpful', 'useless', 'pointless', 'waste of time',
            'obvious', 'already know', 'too simple', 'condescending', 'assuming',
            'wrong', 'incorrect', 'misleading', 'irrelevant', 'inappropriate',
            'bad', 'terrible', 'awful', 'poor', 'stupid', 'dumb', 'ridiculous',
            'disagree', 'reject', 'oppose', 'doubt', 'question', 'suspicious',
            'unreliable', 'untrustworthy', 'inaccurate', 'false', 'untrue',
            'doesn\'t make sense', 'no way', 'that won\'t work', 'not buying it',
            'too vague', 'too broad', 'missing the point', 'off base', 'not convinced'
        ]
        
        # Sentiment modifiers
        intensifiers = ['very', 'extremely', 'highly', 'quite', 'really', 'too', 'so', 
                       'absolutely', 'completely', 'totally', 'definitely', 'certainly']
        negators = ['not', 'no', 'never', 'nothing', 'nowhere', 'nobody', 'none', 
                   'don\'t', 'won\'t', 'can\'t', 'shouldn\'t', 'wouldn\'t', 'isn\'t', 'aren\'t']
        
        # Analyze sentiment
        score = 0
        positive_count = 0
        negative_count = 0
        detected_phrases = []
        intensifier_multiplier = 1.0
        negated = False
        
        # Check for trust-building phrases
        for phrase in trust_builders:
            if phrase in text_lower:
                phrase_score = 2.0 * intensifier_multiplier
                if negated:
                    phrase_score = -phrase_score
                score += phrase_score
                positive_count += 1 if phrase_score > 0 else 0
                negative_count += 1 if phrase_score < 0 else 0
                detected_phrases.append({'phrase': phrase, 'type': 'trust_builder', 'score': phrase_score})
        
        # Check for trust-killing phrases
        for phrase in trust_killers:
            if phrase in text_lower:
                phrase_score = -2.5 * intensifier_multiplier
                if negated:
                    phrase_score = -phrase_score
                score += phrase_score
                positive_count += 1 if phrase_score > 0 else 0
                negative_count += 1 if phrase_score < 0 else 0
                detected_phrases.append({'phrase': phrase, 'type': 'trust_killer', 'score': phrase_score})
        
        # Individual word analysis for missed sentiment
        for i, word in enumerate(words):
            # Check for modifiers
            if word in intensifiers:
                intensifier_multiplier = 1.5
                continue
            if word in negators:
                negated = True
                continue
                
            # Simple positive/negative words
            positive_words = ['good', 'helpful', 'useful', 'clear', 'right', 'correct', 'yes', 'ok', 'okay']
            negative_words = ['bad', 'wrong', 'unclear', 'confusing', 'no', 'nope', 'incorrect']
            
            if word in positive_words:
                word_score = 1.0 * intensifier_multiplier
                if negated:
                    word_score = -word_score
                score += word_score
                positive_count += 1 if word_score > 0 else 0
                negative_count += 1 if word_score < 0 else 0
            elif word in negative_words:
                word_score = -1.0 * intensifier_multiplier
                if negated:
                    word_score = -word_score
                score += word_score
                positive_count += 1 if word_score > 0 else 0
                negative_count += 1 if word_score < 0 else 0
            
            # Reset modifiers after each content word
            intensifier_multiplier = 1.0
            negated = False
        
        # Normalize score (similar to Claude's approach)
        word_count = len(words)
        if word_count > 0:
            #normalized_score = score / np.sqrt(word_count)
            normalized_score = score / max(np.log(word_count + 1), 1)
        else:
            normalized_score = 0
        
        # Determine sentiment label with more sensitive thresholds
        label = 'neutral'
        confidence = min(abs(normalized_score), 1.0)
        
        # More sensitive thresholds than before
        if normalized_score > 0.25:
            label = 'positive'
        elif normalized_score < -0.25:
            label = 'negative'
        
        if normalized_score > 0.75:
            label = 'very positive'
        elif normalized_score < -0.75:
            label = 'very negative'
        
        return {
            'score': normalized_score,
            'label': label,
            'confidence': confidence,
            'wordCount': word_count,
            'positiveWords': positive_count,
            'negativeWords': negative_count,
            'phrases': detected_phrases,
            'rawScore': score
        }
    
    def analyze_experience_impact(self):
        """Analyze how experience level affects sentiment and trust."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        print(f"\nEXPERIENCE LEVEL IMPACT ANALYSIS")
        print("=" * 50)
        
        # Group by experience level
        experience_groups = self.valid_interactions.groupby('Experience')
        
        print("Experience Level Distribution:")
        exp_counts = self.valid_interactions['Experience'].value_counts()
        for exp, count in exp_counts.items():
            print(f"  {exp}: {count} interactions ({count/len(self.valid_interactions)*100:.1f}%)")
        
        print(f"\nDetailed Analysis by Experience Level:")
        print("-" * 60)
        
        experience_stats = {}
        
        for experience, group in experience_groups:
            n = len(group)
            avg_sentiment = group['sentiment_score'].mean()
            avg_trust = group['trust_score'].mean()
            sentiment_std = group['sentiment_score'].std()
            trust_std = group['trust_score'].std()
            
            # Sentiment distribution
            sentiment_dist = group['sentiment_label'].value_counts()
            
            # High/low trust percentages
            high_trust_pct = (group['trust_score'] >= 4).mean() * 100
            low_trust_pct = (group['trust_score'] <= 2).mean() * 100
            
            experience_stats[experience] = {
                'n': n,
                'avg_sentiment': avg_sentiment,
                'avg_trust': avg_trust,
                'sentiment_std': sentiment_std,
                'trust_std': trust_std,
                'high_trust_pct': high_trust_pct,
                'low_trust_pct': low_trust_pct,
                'sentiment_dist': sentiment_dist
            }
            
            print(f"\n{experience.upper()} (N = {n}):")
            print(f"  Trust Performance:")
            print(f"    Average Trust: {avg_trust:.3f} (±{trust_std:.3f})")
            print(f"    High Trust (4-5): {high_trust_pct:.1f}%")
            print(f"    Low Trust (1-2): {low_trust_pct:.1f}%")
            
            print(f"  Sentiment Response:")
            print(f"    Average Sentiment: {avg_sentiment:.3f} (±{sentiment_std:.3f})")
            print(f"    Sentiment Distribution:")
            
            for sentiment_type in ['very positive', 'positive', 'neutral', 'negative', 'very negative']:
                count = sentiment_dist.get(sentiment_type, 0)
                pct = (count / n) * 100 if n > 0 else 0
                print(f"      {sentiment_type.title()}: {count} ({pct:.1f}%)")
        
        # Statistical comparison
        print(f"\nSTATISTICAL INSIGHTS:")
        print("-" * 30)
        
        # Sort by trust level
        sorted_by_trust = sorted(experience_stats.items(), 
                               key=lambda x: x[1]['avg_trust'], reverse=True)
        
        print("Experience Levels Ranked by Trust:")
        for i, (exp, stats) in enumerate(sorted_by_trust, 1):
            print(f"  {i}. {exp}: {stats['avg_trust']:.3f} trust, {stats['avg_sentiment']:.3f} sentiment")
        
        # Identify patterns
        highest_trust_exp = sorted_by_trust[0][0]
        highest_sentiment_exp = max(experience_stats.items(), 
                                  key=lambda x: x[1]['avg_sentiment'])[0]
        most_neutral_exp = max(experience_stats.items(),
                             key=lambda x: x[1]['sentiment_dist'].get('neutral', 0))[0]
        
        print(f"\nKey Patterns:")
        print(f"• Highest Trust: {highest_trust_exp} ({experience_stats[highest_trust_exp]['avg_trust']:.3f})")
        print(f"• Highest Sentiment: {highest_sentiment_exp} ({experience_stats[highest_sentiment_exp]['avg_sentiment']:.3f})")
        print(f"• Most Neutral Responses: {most_neutral_exp}")
        
        # Correlation between experience and sentiment/trust
        exp_order = {'Novice': 1, 'Awareness': 2, 'Intermediate': 3, 'Advanced': 4, 'Expert': 5}
        
        if all(exp in exp_order for exp in self.valid_interactions['Experience'].unique()):
            self.valid_interactions['experience_numeric'] = self.valid_interactions['Experience'].map(exp_order)
            
            exp_sentiment_corr = self.valid_interactions['experience_numeric'].corr(self.valid_interactions['sentiment_score'])
            exp_trust_corr = self.valid_interactions['experience_numeric'].corr(self.valid_interactions['trust_score'])
            
            print(f"\nExperience Level Correlations:")
            print(f"• Experience vs Sentiment: r = {exp_sentiment_corr:.3f}")
            print(f"• Experience vs Trust: r = {exp_trust_corr:.3f}")
        
        return experience_stats
    
    def create_experience_visualizations(self):
        """Create visualizations for experience level analysis."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experience Level Impact on Sentiment and Trust', fontsize=16, fontweight='bold')
        
        # 1. Average sentiment by experience
        exp_sentiment = self.valid_interactions.groupby('Experience')['sentiment_score'].mean().sort_values(ascending=False)
        exp_sentiment.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Average Sentiment Score by Experience Level')
        axes[0,0].set_ylabel('Average Sentiment Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Average trust by experience
        exp_trust = self.valid_interactions.groupby('Experience')['trust_score'].mean().sort_values(ascending=False)
        exp_trust.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Average Trust Score by Experience Level')
        axes[0,1].set_ylabel('Average Trust Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Sentiment distribution by experience
        sentiment_by_exp = pd.crosstab(self.valid_interactions['Experience'], 
                                     self.valid_interactions['sentiment_label'], 
                                     normalize='index') * 100
        sentiment_by_exp.plot(kind='bar', stacked=True, ax=axes[1,0])
        axes[1,0].set_title('Sentiment Distribution by Experience Level (%)')
        axes[1,0].set_ylabel('Percentage')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Sentiment', bbox_to_anchor=(1.05, 1))
        
        # 4. Trust vs Sentiment by experience
        for exp in self.valid_interactions['Experience'].unique():
            exp_data = self.valid_interactions[self.valid_interactions['Experience'] == exp]
            axes[1,1].scatter(exp_data['sentiment_score'], exp_data['trust_score'], 
                            label=exp, alpha=0.6)
        
        axes[1,1].set_title('Sentiment vs Trust by Experience Level')
        axes[1,1].set_xlabel('Sentiment Score')
        axes[1,1].set_ylabel('Trust Score')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experience_level_analysis.png', dpi=300, bbox_inches='tight')
        print("Experience level analysis plots saved as 'experience_level_analysis.png'")
        plt.show()

    def analyze_llm_communication_style(self, text):
        """Analyze LLM communication characteristics."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return self._empty_style_analysis()
        
        clean_text = text.strip()
        words = clean_text.split()
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'length': len(clean_text),
            'wordCount': len(words),
            'sentenceCount': len(sentences),
            'avgWordsPerSentence': len(words) / len(sentences) if sentences else 0,
            'isQuestion': clean_text.strip().endswith('?'),
            'questionCount': clean_text.count('?'),
            'hasNumbers': bool(re.search(r'\d', clean_text)),
            
            # Communication style patterns
            'startsWithQuestion': bool(re.match(r'^(would|have|could|should|can|do|did|will|what|how|why|when|where)\b', clean_text, re.I)),
            'hasTechnicalTerms': bool(re.search(r'\b(pressure|flow|temperature|valve|pump|injection|production|reservoir|wellhead|tubing|gas|oil|water|GOR|survey|analysis|model|acoustic|membrane|separator|choke|flowline|manifold|subsea|downhole|completion|perforation|stimulation|fracturing|acidizing|workover|intervention|logging|drilling|casing|annulus|formation|permeability|porosity|saturation|viscosity|density|API|WOR|PI|IPR|separator|compressor|turbine|generator|transformer|controller|sensor|transmitter|gauge|meter|indicator|alarm|shutdown|trip|interlock|safety|SCADA|DCS|PLC|monitoring|control|automation|optimization|performance|efficiency|reliability|availability|maintainability|troubleshooting|diagnostics|root cause|failure|incident|maintenance|inspection|testing|commissioning|startup|operational|design|engineering|specification|procedure|standard|guideline|recommendation|best practice)\b', clean_text, re.I)),
            
            # Confidence/uncertainty markers
            'hasUncertainty': bool(re.search(r'\b(might|could|maybe|perhaps|possibly|probably|likely|uncertain|unsure|may|seems|appears|looks like|suggests|indicates|potentially|conceivably)\b', clean_text, re.I)),
            'hasConfidence': bool(re.search(r'\b(definitely|certainly|clearly|obviously|sure|confirm|establish|determine|will|should|must|absolutely|undoubtedly|without doubt|for certain)\b', clean_text, re.I)),
            
            # Communication tone
            'isDirective': bool(re.search(r'\b(should|must|need to|have to|require|recommend|suggest|advise|propose|urge|insist|demand)\b', clean_text, re.I)),
            'isCollaborative': bool(re.search(r'\b(we|us|our|together|work with|help|assist|support|collaborate|partner|team up|join forces)\b', clean_text, re.I)),
            'isExplorative': bool(re.search(r'\b(explore|investigate|check|examine|look into|consider|review|analyze|study|research|evaluate|assess|test|verify|validate)\b', clean_text, re.I)),
            
            'text': clean_text
        }
    
    def _empty_style_analysis(self):
        """Return empty analysis for missing text."""
        return {
            'length': 0, 'wordCount': 0, 'sentenceCount': 0, 'avgWordsPerSentence': 0,
            'isQuestion': False, 'questionCount': 0, 'hasNumbers': False,
            'startsWithQuestion': False, 'hasTechnicalTerms': False,
            'hasUncertainty': False, 'hasConfidence': False, 'isDirective': False,
            'isCollaborative': False, 'isExplorative': False, 'text': ''
        }
    
    def process_all_data(self):
        """Process all interactions with proper filtering."""
        processed_rows = []
        valid_response_count = 0
        total_interactions = 0
        
        for idx, row in self.data.iterrows():
            for q in range(1, 6):  # Q1 through Q5
                total_interactions += 1
                
                # Get the data (handle different naming conventions)
                user_response = row.get(f'Q{q}resp', '')
                trust_score = row.get(f'Q{q}_Trust')
                if pd.isna(trust_score):
                    trust_score = row.get(f'Q{q}_trust')
                llm_text = row.get(f'Q{q}_LLM', '')
                
                # Only process if we have a trust score
                if pd.notna(trust_score):
                    # Check if user response is valid
                    has_valid_response = self.is_valid_response(user_response)
                    if has_valid_response:
                        valid_response_count += 1
                    
                    # Analyze sentiment (only for valid responses)
                    sentiment = self.analyze_sentiment(user_response) if has_valid_response else {
                        'score': 0, 'label': 'no_response', 'confidence': 0, 'wordCount': 0,
                        'positiveWords': 0, 'negativeWords': 0, 'phrases': []
                    }
                    
                    # Analyze LLM communication style
                    llm_style = self.analyze_llm_communication_style(llm_text)
                    
                    processed_row = {
                        'participant_id': idx + 1,
                        'LLM': row['LLM'],
                        'Experience': row['Experience'],
                        'question': q,
                        'trust_score': trust_score,
                        'user_response': user_response if has_valid_response else '',
                        'llm_text': llm_text,
                        'has_valid_response': has_valid_response,
                        
                        # Sentiment analysis results
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'sentiment_confidence': sentiment['confidence'],
                        'sentiment_word_count': sentiment['wordCount'],
                        'sentiment_positive_words': sentiment['positiveWords'],
                        'sentiment_negative_words': sentiment['negativeWords'],
                        'sentiment_phrases_detected': len(sentiment['phrases']),
                        
                        # LLM style analysis results  
                        'llm_length': llm_style['length'],
                        'llm_word_count': llm_style['wordCount'],
                        'llm_sentence_count': llm_style['sentenceCount'],
                        'llm_avg_words_per_sentence': llm_style['avgWordsPerSentence'],
                        'llm_is_question': llm_style['isQuestion'],
                        'llm_question_count': llm_style['questionCount'],
                        'llm_has_numbers': llm_style['hasNumbers'],
                        'llm_starts_with_question': llm_style['startsWithQuestion'],
                        'llm_has_technical_terms': llm_style['hasTechnicalTerms'],
                        'llm_has_uncertainty': llm_style['hasUncertainty'],
                        'llm_has_confidence': llm_style['hasConfidence'],
                        'llm_is_directive': llm_style['isDirective'],
                        'llm_is_collaborative': llm_style['isCollaborative'],
                        'llm_is_explorative': llm_style['isExplorative']
                    }
                    processed_rows.append(processed_row)
        
        self.processed_data = pd.DataFrame(processed_rows)
        self.valid_interactions = self.processed_data[self.processed_data['has_valid_response']].copy()
        
        print(f"Processed {len(self.processed_data)} total interactions")
        print(f"Valid responses with text: {len(self.valid_interactions)} ({len(self.valid_interactions)/len(self.processed_data)*100:.1f}%)")
        print(f"Empty/undefined responses filtered out: {len(self.processed_data) - len(self.valid_interactions)}")
        
        return self.processed_data
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        if self.processed_data is None:
            self.process_all_data()
        
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS SUMMARY (Matching Claude's JavaScript)")
        print("="*60)
        
        # Overall statistics
        total_interactions = len(self.processed_data)
        valid_responses = len(self.valid_interactions)
        
        print(f"Total interactions: {total_interactions}")
        print(f"Valid responses: {valid_responses} ({valid_responses/total_interactions*100:.1f}%)")
        print(f"Filtered out: {total_interactions - valid_responses} empty/undefined responses")
        
        # Sentiment distribution (only valid responses)
        print(f"\nSentiment Distribution (Valid Responses Only):")
        sentiment_dist = self.valid_interactions['sentiment_label'].value_counts()
        for label, count in sentiment_dist.items():
            print(f"  {label}: {count} ({count/len(self.valid_interactions)*100:.1f}%)")
        
        # LLM performance (valid responses only)
        print(f"\nLLM Performance Summary (Valid Responses Only):")
        llm_stats = self.valid_interactions.groupby('LLM').agg({
            'trust_score': ['mean', 'std', 'count'],
            'sentiment_score': ['mean', 'std'],
            'llm_length': 'mean',
            'llm_question_count': 'mean',
            'llm_has_technical_terms': 'mean',
            'llm_is_question': 'mean',
            'llm_has_uncertainty': 'mean',
            'llm_has_confidence': 'mean'
        }).round(3)
        
        print(llm_stats)
        
        # Key correlations (valid responses only)
        print(f"\nKey Correlations (Valid Responses Only):")
        correlations = {}
        try:
            correlations['Sentiment-Trust'] = pearsonr(self.valid_interactions['sentiment_score'], 
                                                     self.valid_interactions['trust_score'])[0]
            correlations['Length-Trust'] = pearsonr(self.valid_interactions['llm_length'], 
                                                   self.valid_interactions['trust_score'])[0]
            correlations['Questions-Trust'] = pearsonr(self.valid_interactions['llm_question_count'], 
                                                     self.valid_interactions['trust_score'])[0]
            correlations['Uncertainty-Trust'] = pearsonr(self.valid_interactions['llm_has_uncertainty'].astype(int), 
                                                        self.valid_interactions['trust_score'])[0]
            correlations['Confidence-Trust'] = pearsonr(self.valid_interactions['llm_has_confidence'].astype(int), 
                                                       self.valid_interactions['trust_score'])[0]
        except Exception as e:
            print(f"Error calculating correlations: {e}")
        
        for name, corr in correlations.items():
            print(f"  {name}: r = {corr:.3f}")
        
        # Trust score distribution by LLM
        print(f"\nTrust Score Distribution by LLM:")
        for llm in sorted(self.valid_interactions['LLM'].unique()):
            llm_data = self.valid_interactions[self.valid_interactions['LLM'] == llm]
            trust_dist = llm_data['trust_score'].value_counts().sort_index()
            high_trust = ((llm_data['trust_score'] >= 4).sum() / len(llm_data) * 100)
            low_trust = ((llm_data['trust_score'] <= 2).sum() / len(llm_data) * 100)
            print(f"  {llm}: High Trust (4-5): {high_trust:.1f}%, Low Trust (1-2): {low_trust:.1f}%")
        
        return llm_stats, correlations
    
    def create_enhanced_visualizations(self, save_plots=True):
        """Create enhanced visualizations matching Claude's insights."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        # Set up plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Enhanced LLM Communication-Trust Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Trust distribution by LLM (stacked bar)
        trust_pivot = self.valid_interactions.pivot_table(
            values='participant_id', index='LLM', columns='trust_score', 
            aggfunc='count', fill_value=0
        )
        trust_pivot.plot(kind='bar', stacked=True, ax=axes[0,0], colormap='RdYlGn')
        axes[0,0].set_title('Trust Score Distribution by LLM')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend(title='Trust Score', bbox_to_anchor=(1.05, 1))
        
        # 2. Sentiment vs Trust scatter (colored by LLM)
        for llm in self.valid_interactions['LLM'].unique():
            llm_data = self.valid_interactions[self.valid_interactions['LLM'] == llm]
            axes[0,1].scatter(llm_data['sentiment_score'], llm_data['trust_score'], 
                            label=llm, alpha=0.7, s=50)
        axes[0,1].set_title('Sentiment vs Trust Score')
        axes[0,1].set_xlabel('Sentiment Score')
        axes[0,1].set_ylabel('Trust Score')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. LLM response length distribution
        self.valid_interactions.boxplot(column='llm_length', by='LLM', ax=axes[0,2])
        axes[0,2].set_title('LLM Response Length Distribution')
        axes[0,2].set_ylabel('Character Count')
        
        # 4. Average metrics by LLM
        metrics = ['trust_score', 'sentiment_score']
        llm_means = self.valid_interactions.groupby('LLM')[metrics].mean()
        llm_means.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Average Trust & Sentiment by LLM')
        axes[1,0].set_ylabel('Score')
        axes[1,0].tick_params(axis='x', rotation=0)
        axes[1,0].legend()
        
        # 5. Sentiment distribution pie chart
        sentiment_counts = self.valid_interactions['sentiment_label'].value_counts()
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold', 'lightsalmon']
        axes[1,1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                     colors=colors[:len(sentiment_counts)])
        axes[1,1].set_title('Sentiment Distribution\n(Valid Responses)')
        
        # 6. Communication style heatmap
        style_features = ['llm_is_question', 'llm_has_technical_terms', 'llm_has_uncertainty', 
                         'llm_has_confidence', 'llm_is_directive', 'llm_is_explorative']
        style_by_llm = self.valid_interactions.groupby('LLM')[style_features].mean()
        sns.heatmap(style_by_llm.T, annot=True, fmt='.2f', ax=axes[1,2], cmap='YlOrRd')
        axes[1,2].set_title('Communication Style Patterns')
        
        # 7. Trust score trends
        trust_means = self.valid_interactions.groupby('LLM')['trust_score'].mean().sort_values(ascending=False)
        trust_means.plot(kind='bar', ax=axes[2,0], color='steelblue')
        axes[2,0].set_title('LLM Trust Ranking')
        axes[2,0].set_ylabel('Average Trust Score')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # 8. Question rate vs Trust
        question_rate = self.valid_interactions.groupby('LLM')['llm_is_question'].mean() * 100
        trust_avg = self.valid_interactions.groupby('LLM')['trust_score'].mean()
        axes[2,1].scatter(question_rate, trust_avg, s=100, alpha=0.7)
        for llm in question_rate.index:
            axes[2,1].annotate(llm, (question_rate[llm], trust_avg[llm]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[2,1].set_title('Question Rate vs Average Trust')
        axes[2,1].set_xlabel('Question Rate (%)')
        axes[2,1].set_ylabel('Average Trust Score')
        axes[2,1].grid(True, alpha=0.3)
        
        # 9. High vs Low Trust comparison
        high_trust = self.valid_interactions[self.valid_interactions['trust_score'] >= 4]
        low_trust = self.valid_interactions[self.valid_interactions['trust_score'] <= 2]
        
        trust_comparison = pd.DataFrame({
            'High Trust (4-5)': [high_trust['sentiment_score'].mean(), 
                                high_trust['llm_length'].mean(),
                                high_trust['llm_is_question'].mean() * 100],
            'Low Trust (1-2)': [low_trust['sentiment_score'].mean(),
                               low_trust['llm_length'].mean(), 
                               low_trust['llm_is_question'].mean() * 100]
        }, index=['Avg Sentiment', 'Avg Length', 'Question Rate (%)'])
        
        trust_comparison.plot(kind='bar', ax=axes[2,2])
        axes[2,2].set_title('High vs Low Trust Characteristics')
        axes[2,2].tick_params(axis='x', rotation=45)
        axes[2,2].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('enhanced_llm_analysis_dashboard.png', dpi=300, bbox_inches='tight')
            print(f"\nEnhanced dashboard saved as 'enhanced_llm_analysis_dashboard.png'")
        
        plt.show()
    
    def build_enhanced_prediction_model(self):
        """Build enhanced trust prediction model."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        print(f"\nBuilding Trust Prediction Model on {len(self.valid_interactions)} valid interactions...")
        
        # Feature engineering
        feature_columns = [
            'llm_length', 'llm_word_count', 'llm_question_count',
            'llm_has_technical_terms', 'llm_has_uncertainty', 'llm_has_confidence',
            'llm_is_directive', 'llm_is_collaborative', 'llm_is_explorative',
            'llm_is_question', 'llm_starts_with_question',
            'sentiment_score', 'sentiment_confidence', 'sentiment_positive_words', 'sentiment_negative_words'
        ]
        
        # Prepare data
        model_data = self.valid_interactions.copy()
        
        # Convert boolean columns to int
        for col in feature_columns:
            if model_data[col].dtype == bool:
                model_data[col] = model_data[col].astype(int)
        
        # Handle any missing values
        model_data[feature_columns] = model_data[feature_columns].fillna(0)
        
        X = model_data[feature_columns]
        y = model_data['trust_score']
        
        print(f"Features used: {len(feature_columns)}")
        print(f"Training samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        print(f"\nModel Performance:")
        print("-" * 40)
        
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
        
        return results, X_test, y_test, feature_columns
    
    def analyze_phrase_patterns(self):
        """Analyze specific phrase patterns that drive trust."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        print(f"\nPHRASE-LEVEL TRUST ANALYSIS")
        print("=" * 40)
        
        # Collect all phrases detected in sentiment analysis
        trust_builder_phrases = {}
        trust_killer_phrases = {}
        
        for _, row in self.valid_interactions.iterrows():
            # Analyze common words in high vs low trust responses
            if row['trust_score'] >= 4:  # High trust
                words = str(row['user_response']).lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        trust_builder_phrases[word] = trust_builder_phrases.get(word, 0) + 1
            elif row['trust_score'] <= 2:  # Low trust
                words = str(row['user_response']).lower().split()
                for word in words:
                    if len(word) > 3:
                        trust_killer_phrases[word] = trust_killer_phrases.get(word, 0) + 1
        
        print("Top Trust-Building Words/Phrases:")
        for word, count in sorted(trust_builder_phrases.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count >= 2:  # Only show words that appear multiple times
                avg_trust = self.valid_interactions[
                    self.valid_interactions['user_response'].str.contains(word, case=False, na=False)
                ]['trust_score'].mean()
                print(f"  '{word}': {count} occurrences, avg trust: {avg_trust:.2f}")
        
        print("\nTop Trust-Killing Words/Phrases:")
        for word, count in sorted(trust_killer_phrases.items(), key=lambda x: x[1], reverse=True)[:10]:
            if count >= 2:
                avg_trust = self.valid_interactions[
                    self.valid_interactions['user_response'].str.contains(word, case=False, na=False)
                ]['trust_score'].mean()
                print(f"  '{word}': {count} occurrences, avg trust: {avg_trust:.2f}")
    
    def generate_llm_communication_profiles(self):
        """Generate detailed communication profiles for each LLM."""
        if self.valid_interactions is None:
            self.process_all_data()
        
        print(f"\nLLM COMMUNICATION PROFILES")
        print("=" * 50)
        
        for llm in sorted(self.valid_interactions['LLM'].unique()):
            llm_data = self.valid_interactions[self.valid_interactions['LLM'] == llm]
            
            print(f"\n{llm} (N = {len(llm_data)}):")
            print(f"  Trust Performance:")
            print(f"    Average Trust: {llm_data['trust_score'].mean():.3f}")
            print(f"    Trust Std Dev: {llm_data['trust_score'].std():.3f}")
            print(f"    High Trust (4-5): {(llm_data['trust_score'] >= 4).sum()}/{len(llm_data)} ({(llm_data['trust_score'] >= 4).mean()*100:.1f}%)")
            print(f"    Low Trust (1-2): {(llm_data['trust_score'] <= 2).sum()}/{len(llm_data)} ({(llm_data['trust_score'] <= 2).mean()*100:.1f}%)")
            
            print(f"  Communication Style:")
            print(f"    Avg Length: {llm_data['llm_length'].mean():.1f} characters")
            print(f"    Avg Words: {llm_data['llm_word_count'].mean():.1f}")
            print(f"    Questions: {llm_data['llm_is_question'].mean()*100:.1f}%")
            print(f"    Technical Terms: {llm_data['llm_has_technical_terms'].mean()*100:.1f}%")
            print(f"    Uncertainty Language: {llm_data['llm_has_uncertainty'].mean()*100:.1f}%")
            print(f"    Confidence Language: {llm_data['llm_has_confidence'].mean()*100:.1f}%")
            print(f"    Directive Tone: {llm_data['llm_is_directive'].mean()*100:.1f}%")
            print(f"    Collaborative Tone: {llm_data['llm_is_collaborative'].mean()*100:.1f}%")
            print(f"    Explorative Tone: {llm_data['llm_is_explorative'].mean()*100:.1f}%")
            
            print(f"  User Sentiment Response:")
            print(f"    Average Sentiment: {llm_data['sentiment_score'].mean():.3f}")
            print(f"    Positive Responses: {(llm_data['sentiment_label'] == 'positive').sum()}/{len(llm_data)} ({(llm_data['sentiment_label'] == 'positive').mean()*100:.1f}%)")
            print(f"    Negative Responses: {(llm_data['sentiment_label'] == 'negative').sum()}/{len(llm_data)} ({(llm_data['sentiment_label'] == 'negative').mean()*100:.1f}%)")
            
            # Show a few examples
            high_trust_examples = llm_data[llm_data['trust_score'] >= 4].head(2)
            low_trust_examples = llm_data[llm_data['trust_score'] <= 2].head(2)
            
            if len(high_trust_examples) > 0:
                print(f"  High Trust Example:")
                example = high_trust_examples.iloc[0]
                print(f"    LLM: \"{example['llm_text'][:100]}...\"")
                print(f"    User: \"{example['user_response'][:80]}...\" (Trust: {example['trust_score']})")
            
            if len(low_trust_examples) > 0:
                print(f"  Low Trust Example:")
                example = low_trust_examples.iloc[0]
                print(f"    LLM: \"{example['llm_text'][:100]}...\"")
                print(f"    User: \"{example['user_response'][:80]}...\" (Trust: {example['trust_score']})")
    
    def export_enhanced_results(self, filename='enhanced_llm_analysis_results.csv'):
        """Export enhanced processed data and results to CSV."""
        if self.processed_data is None:
            self.process_all_data()
        
        # Export full dataset
        self.processed_data.to_csv(filename, index=False)
        
        # Export summary statistics
        summary_filename = filename.replace('.csv', '_summary.csv')
        if self.valid_interactions is not None:
            summary_stats = self.valid_interactions.groupby('LLM').agg({
                'trust_score': ['mean', 'std', 'count'],
                'sentiment_score': ['mean', 'std'],
                'sentiment_positive_words': 'sum',
                'sentiment_negative_words': 'sum',
                'llm_length': 'mean',
                'llm_question_count': 'mean',
                'llm_is_question': 'mean',
                'llm_has_technical_terms': 'mean',
                'llm_has_uncertainty': 'mean',
                'llm_has_confidence': 'mean'
            }).round(3)
            
            summary_stats.to_csv(summary_filename)
            
        print(f"Enhanced results exported to:")
        print(f"  - {filename}: Complete dataset")
        print(f"  - {summary_filename}: Summary statistics")

def main():
    """Main execution function."""
    print("Enhanced LLM Communication-Trust Analysis")
    print("Replicating Claude's JavaScript analysis with proper data filtering")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = EnhancedLLMTrustAnalyzer('LLM_Comm_trust_Resp.csv')
    
    # Process data with proper filtering
    processed_data = analyzer.process_all_data()
    
    # Generate summary statistics
    llm_stats, correlations = analyzer.generate_summary_statistics()
    
    # Generate detailed LLM profiles
    analyzer.generate_llm_communication_profiles()
    
    # Analyze phrase patterns
    analyzer.analyze_phrase_patterns()
    
    # Create enhanced visualizations
    analyzer.create_enhanced_visualizations()
    
    # Build prediction model
    model_results, X_test, y_test, features = analyzer.build_enhanced_prediction_model()
    
    # Add after the existing analyses, before export_results()
    analyzer.analyze_experience_impact()
    analyzer.create_experience_visualizations()

    # Export results
    analyzer.export_enhanced_results()
    
    print("\n" + "="*70)
    print("ENHANCED ANALYSIS COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("- enhanced_llm_analysis_dashboard.png: Comprehensive visual dashboard")
    print("- enhanced_llm_analysis_results.csv: Complete processed dataset")
    print("- enhanced_llm_analysis_results_summary.csv: Summary statistics by LLM")
    print("\nThis analysis should now match Claude's JavaScript results much more closely!")
    print("Key improvements:")
    print("- Proper filtering of empty/undefined responses")
    print("- Enhanced sentiment analysis with trust-specific lexicon")
    print("- More sensitive sentiment thresholds")
    print("- Comprehensive communication style analysis")

if __name__ == "__main__":
    main()