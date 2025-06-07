#!/usr/bin/env python3
"""
Detailed Dental Results Analysis

This script provides additional detailed analysis of the dental AI model results,
including error pattern analysis, statistical tests, and question categorization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
from scipy import stats

class DetailedDentalAnalyzer:
    def __init__(self, results_dir='results/dental'):
        """Initialize the detailed analyzer."""
        self.results_dir = Path(results_dir)
        self.models = {}
        self.comparison_data = None
        
    def load_data(self):
        """Load CSV files for all models."""
        csv_files = {
            'AGIR': 'agir_results.csv',
            'GPT-4.1-nano': 'gpt-4.1-nano_dental_results.csv', 
            'O3-mini': 'o3-mini_dental_results.csv'
        }
        
        for model_name, filename in csv_files.items():
            file_path = self.results_dir / filename
            if file_path.exists():
                self.models[model_name] = pd.read_csv(file_path)
                # Standardize is_correct column
                if 'is_correct' in self.models[model_name].columns:
                    df = self.models[model_name]
                    if df['is_correct'].dtype == 'object':
                        df['is_correct'] = df['is_correct'].map({
                            'True': True, 'False': False, 
                            1: True, 0: False, 
                            '1': True, '0': False
                        })
                    else:
                        df['is_correct'] = df['is_correct'].astype(bool)
        
        # Create comparison dataframe
        if len(self.models) >= 2:
            comparison_data = []
            question_sets = [set(df['question_id']) for df in self.models.values()]
            common_questions = set.intersection(*question_sets)
            
            for question_id in common_questions:
                row = {'question_id': question_id}
                
                for model_name, df in self.models.items():
                    question_data = df[df['question_id'] == question_id].iloc[0]
                    row[f'{model_name}_correct'] = question_data['is_correct']
                    row[f'{model_name}_answer'] = question_data['predicted_answer']
                    row['question'] = question_data['question']
                    if 'correct_answer' in question_data:
                        row['correct_answer'] = question_data['correct_answer']
                    elif 'correct_option' in question_data:
                        row['correct_answer'] = question_data['correct_option']
                
                comparison_data.append(row)
            
            self.comparison_data = pd.DataFrame(comparison_data)
            
        return self.models
    
    def analyze_error_patterns(self):
        """Analyze common error patterns across models."""
        if self.comparison_data is None:
            return None
        
        model_names = list(self.models.keys())
        error_patterns = {}
        
        # Categorize questions by how many models got them wrong
        correct_cols = [f'{model}_correct' for model in model_names]
        self.comparison_data['models_correct'] = self.comparison_data[correct_cols].sum(axis=1)
        
        # Find questions that all models got wrong
        all_wrong = self.comparison_data[self.comparison_data['models_correct'] == 0]
        
        # Find questions that only one model got right
        one_correct = self.comparison_data[self.comparison_data['models_correct'] == 1]
        
        # Find questions that all models got right
        all_correct = self.comparison_data[self.comparison_data['models_correct'] == len(model_names)]
        
        error_patterns = {
            'all_models_wrong': {
                'count': len(all_wrong),
                'percentage': len(all_wrong) / len(self.comparison_data) * 100,
                'questions': all_wrong['question'].tolist() if len(all_wrong) > 0 else []
            },
            'one_model_correct': {
                'count': len(one_correct),
                'percentage': len(one_correct) / len(self.comparison_data) * 100,
                'questions': one_correct['question'].tolist() if len(one_correct) > 0 else []
            },
            'all_models_correct': {
                'count': len(all_correct),
                'percentage': len(all_correct) / len(self.comparison_data) * 100
            }
        }
        
        return error_patterns
    
    def analyze_answer_distribution(self):
        """Analyze the distribution of predicted answers."""
        answer_distributions = {}
        
        for model_name, df in self.models.items():
            answer_counts = df['predicted_answer'].value_counts()
            answer_distributions[model_name] = {
                'distribution': answer_counts.to_dict(),
                'most_common': answer_counts.index[0] if len(answer_counts) > 0 else None,
                'most_common_count': answer_counts.iloc[0] if len(answer_counts) > 0 else 0,
                'unique_answers': len(answer_counts)
            }
        
        return answer_distributions
    
    def analyze_correct_answer_distribution(self):
        """Analyze the distribution of correct answers."""
        # Use the first model's data to get correct answers (should be same across all)
        first_model = list(self.models.keys())[0]
        df = self.models[first_model]
        
        if 'correct_answer' in df.columns:
            correct_col = 'correct_answer'
        elif 'correct_option' in df.columns:
            correct_col = 'correct_option'
        else:
            return None
        
        correct_distribution = df[correct_col].value_counts()
        
        return {
            'distribution': correct_distribution.to_dict(),
            'most_common_correct': correct_distribution.index[0],
            'most_common_count': correct_distribution.iloc[0],
            'balance_score': correct_distribution.std() / correct_distribution.mean() if correct_distribution.mean() > 0 else 0
        }
    
    def statistical_significance_tests(self):
        """Perform statistical significance tests between models."""
        if len(self.models) < 2:
            return None
        
        model_names = list(self.models.keys())
        results = {}
        
        # McNemar's test for pairwise comparison
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Get common questions
                common_questions = set(self.models[model1]['question_id']) & set(self.models[model2]['question_id'])
                
                df1_filtered = self.models[model1][self.models[model1]['question_id'].isin(common_questions)]
                df2_filtered = self.models[model2][self.models[model2]['question_id'].isin(common_questions)]
                
                # Sort by question_id to ensure same order
                df1_filtered = df1_filtered.sort_values('question_id')
                df2_filtered = df2_filtered.sort_values('question_id')
                
                # Create contingency table for McNemar's test
                model1_correct = df1_filtered['is_correct'].values
                model2_correct = df2_filtered['is_correct'].values
                
                # Contingency table: [both_correct, model1_only, model2_only, both_wrong]
                both_correct = sum((model1_correct == True) & (model2_correct == True))
                model1_only = sum((model1_correct == True) & (model2_correct == False))
                model2_only = sum((model1_correct == False) & (model2_correct == True))
                both_wrong = sum((model1_correct == False) & (model2_correct == False))
                
                # McNemar's test
                if model1_only + model2_only > 0:
                    mcnemar_stat = abs(model1_only - model2_only) / np.sqrt(model1_only + model2_only)
                    p_value = 2 * (1 - stats.norm.cdf(mcnemar_stat))
                else:
                    mcnemar_stat = 0
                    p_value = 1.0
                
                results[f'{model1}_vs_{model2}'] = {
                    'both_correct': both_correct,
                    'model1_only_correct': model1_only,
                    'model2_only_correct': model2_only,
                    'both_wrong': both_wrong,
                    'mcnemar_statistic': mcnemar_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return results
    
    def analyze_question_length_impact(self):
        """Analyze if question length affects model performance."""
        if self.comparison_data is None:
            return None
        
        # Calculate question lengths
        self.comparison_data['question_length'] = self.comparison_data['question'].str.len()
        
        # Categorize by length
        length_quartiles = self.comparison_data['question_length'].quantile([0.25, 0.5, 0.75])
        
        def categorize_length(length):
            if length <= length_quartiles[0.25]:
                return 'Short'
            elif length <= length_quartiles[0.5]:
                return 'Medium-Short'
            elif length <= length_quartiles[0.75]:
                return 'Medium-Long'
            else:
                return 'Long'
        
        self.comparison_data['length_category'] = self.comparison_data['question_length'].apply(categorize_length)
        
        # Analyze performance by length category
        model_names = list(self.models.keys())
        length_analysis = {}
        
        for model in model_names:
            correct_col = f'{model}_correct'
            if correct_col in self.comparison_data.columns:
                performance_by_length = self.comparison_data.groupby('length_category')[correct_col].agg(['count', 'sum', 'mean'])
                performance_by_length['accuracy_percent'] = performance_by_length['mean'] * 100
                length_analysis[model] = performance_by_length.to_dict('index')
        
        return length_analysis
    
    def find_controversial_questions(self, max_examples=10):
        """Find questions where models disagree the most."""
        if self.comparison_data is None:
            return None
        
        model_names = list(self.models.keys())
        correct_cols = [f'{model}_correct' for model in model_names]
        
        # Calculate disagreement score (standard deviation of correct/incorrect across models)
        self.comparison_data['disagreement'] = self.comparison_data[correct_cols].apply(
            lambda row: np.std([int(x) for x in row]), axis=1
        )
        
        # Get most controversial questions
        controversial = self.comparison_data.nlargest(max_examples, 'disagreement')
        
        controversial_questions = []
        for _, row in controversial.iterrows():
            question_info = {
                'question': row['question'],
                'correct_answer': row['correct_answer'],
                'disagreement_score': row['disagreement'],
                'model_responses': {}
            }
            
            for model in model_names:
                question_info['model_responses'][model] = {
                    'answer': row[f'{model}_answer'],
                    'correct': row[f'{model}_correct']
                }
            
            controversial_questions.append(question_info)
        
        return controversial_questions
    
    def generate_detailed_report(self):
        """Generate a comprehensive detailed report."""
        print("\n" + "="*100)
        print("DETAILED DENTAL AI MODELS ANALYSIS REPORT")
        print("="*100)
        
        # Error pattern analysis
        error_patterns = self.analyze_error_patterns()
        if error_patterns:
            print("\n1. ERROR PATTERN ANALYSIS")
            print("-" * 50)
            print(f"Questions all models got wrong: {error_patterns['all_models_wrong']['count']} ({error_patterns['all_models_wrong']['percentage']:.2f}%)")
            print(f"Questions only one model got right: {error_patterns['one_model_correct']['count']} ({error_patterns['one_model_correct']['percentage']:.2f}%)")
            print(f"Questions all models got right: {error_patterns['all_models_correct']['count']} ({error_patterns['all_models_correct']['percentage']:.2f}%)")
        
        # Answer distribution analysis
        answer_distributions = self.analyze_answer_distribution()
        print(f"\n2. ANSWER DISTRIBUTION ANALYSIS")
        print("-" * 50)
        for model, dist in answer_distributions.items():
            print(f"\n{model}:")
            print(f"  Most common predicted answer: {dist['most_common']} ({dist['most_common_count']} times)")
            print(f"  Number of unique answers used: {dist['unique_answers']}")
            print(f"  Answer distribution: {dict(list(dist['distribution'].items())[:5])}")  # Top 5
        
        # Correct answer distribution
        correct_dist = self.analyze_correct_answer_distribution()
        if correct_dist:
            print(f"\n3. CORRECT ANSWER DISTRIBUTION")
            print("-" * 50)
            print(f"Most common correct answer: {correct_dist['most_common_correct']} ({correct_dist['most_common_count']} questions)")
            print(f"Answer distribution balance score: {correct_dist['balance_score']:.3f}")
            print(f"Correct answer distribution: {correct_dist['distribution']}")
        
        # Statistical significance tests
        stat_tests = self.statistical_significance_tests()
        if stat_tests:
            print(f"\n4. STATISTICAL SIGNIFICANCE TESTS")
            print("-" * 50)
            for comparison, result in stat_tests.items():
                print(f"\n{comparison}:")
                print(f"  Both correct: {result['both_correct']}")
                print(f"  Only first model correct: {result['model1_only_correct']}")
                print(f"  Only second model correct: {result['model2_only_correct']}")
                print(f"  Both wrong: {result['both_wrong']}")
                print(f"  McNemar's p-value: {result['p_value']:.4f}")
                print(f"  Statistically significant difference: {'Yes' if result['significant'] else 'No'}")
        
        # Question length impact
        length_analysis = self.analyze_question_length_impact()
        if length_analysis:
            print(f"\n5. QUESTION LENGTH IMPACT ANALYSIS")
            print("-" * 50)
            for model, analysis in length_analysis.items():
                print(f"\n{model} Performance by Question Length:")
                for length_cat, stats in analysis.items():
                    print(f"  {length_cat}: {stats['accuracy_percent']:.2f}% accuracy ({stats['sum']}/{stats['count']})")
        
        # Controversial questions
        controversial = self.find_controversial_questions(5)
        if controversial:
            print(f"\n6. MOST CONTROVERSIAL QUESTIONS (Top 5)")
            print("-" * 50)
            for i, q in enumerate(controversial, 1):
                print(f"\n{i}. Question: {q['question'][:100]}...")
                print(f"   Correct Answer: {q['correct_answer']}")
                print(f"   Disagreement Score: {q['disagreement_score']:.3f}")
                print(f"   Model Responses:")
                for model, response in q['model_responses'].items():
                    status = "✓" if response['correct'] else "✗"
                    print(f"     {model}: {response['answer']} {status}")
        
        print("\n" + "="*100)
    
    def run_detailed_analysis(self):
        """Run the complete detailed analysis."""
        print("Starting Detailed Dental Results Analysis...")
        print("="*60)
        
        self.load_data()
        self.generate_detailed_report()
        
        print("\nDetailed analysis complete!")


def main():
    """Main function to run the detailed analysis."""
    analyzer = DetailedDentalAnalyzer()
    analyzer.run_detailed_analysis()


if __name__ == "__main__":
    main() 