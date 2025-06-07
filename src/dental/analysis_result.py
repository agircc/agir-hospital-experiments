#!/usr/bin/env python3
"""
Dental Results Analysis

This script analyzes the performance results of different AI models on dental questions.
It compares AGIR, GPT-4.1-nano, and O3-mini models across various metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DentalResultsAnalyzer:
    def __init__(self, results_dir='results/dental'):
        """Initialize the analyzer with path to results directory."""
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
                print(f"Loading {model_name} data from {filename}...")
                self.models[model_name] = pd.read_csv(file_path)
                print(f"  {model_name}: {len(self.models[model_name])} questions loaded")
            else:
                print(f"Warning: {filename} not found!")
        
        return self.models
    
    def preprocess_data(self):
        """Standardize and preprocess the data across models."""
        for model_name, df in self.models.items():
            # Standardize column names and data types
            if 'is_correct' in df.columns:
                # Convert boolean values to standardized format
                if df['is_correct'].dtype == 'object':
                    df['is_correct'] = df['is_correct'].map({
                        'True': True, 'False': False, 
                        1: True, 0: False, 
                        '1': True, '0': False
                    })
                else:
                    df['is_correct'] = df['is_correct'].astype(bool)
            
            # Add model name column
            df['model'] = model_name
            
            print(f"{model_name} data shape: {df.shape}")
    
    def calculate_basic_metrics(self):
        """Calculate basic performance metrics for each model."""
        metrics = {}
        
        for model_name, df in self.models.items():
            total_questions = len(df)
            correct_answers = df['is_correct'].sum()
            accuracy = correct_answers / total_questions if total_questions > 0 else 0
            
            metrics[model_name] = {
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'incorrect_answers': total_questions - correct_answers,
                'accuracy': accuracy,
                'accuracy_percent': accuracy * 100
            }
        
        return metrics
    
    def create_comparison_dataframe(self):
        """Create a unified dataframe for comparison across models."""
        comparison_data = []
        
        # Get common question IDs across all models
        if len(self.models) >= 2:
            question_sets = [set(df['question_id']) for df in self.models.values()]
            common_questions = set.intersection(*question_sets)
            print(f"Common questions across all models: {len(common_questions)}")
            
            for question_id in common_questions:
                row = {'question_id': question_id}
                
                for model_name, df in self.models.items():
                    question_data = df[df['question_id'] == question_id].iloc[0]
                    row[f'{model_name}_correct'] = question_data['is_correct']
                    row[f'{model_name}_answer'] = question_data['predicted_answer']
                    if 'correct_answer' in question_data:
                        row['correct_answer'] = question_data['correct_answer']
                    elif 'correct_option' in question_data:
                        row['correct_answer'] = question_data['correct_option']
                
                comparison_data.append(row)
            
            self.comparison_data = pd.DataFrame(comparison_data)
            return self.comparison_data
        else:
            print("Need at least 2 models for comparison")
            return None
    
    def analyze_agreement(self):
        """Analyze agreement between models."""
        if self.comparison_data is None:
            print("No comparison data available")
            return None
        
        model_names = list(self.models.keys())
        agreement_stats = {}
        
        if len(model_names) >= 2:
            # Pairwise agreement
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    col1 = f'{model1}_correct'
                    col2 = f'{model2}_correct'
                    
                    if col1 in self.comparison_data.columns and col2 in self.comparison_data.columns:
                        agreement = (self.comparison_data[col1] == self.comparison_data[col2]).mean()
                        agreement_stats[f'{model1}_vs_{model2}'] = agreement * 100
            
            # All models agreement
            if len(model_names) == 3:
                all_correct_cols = [f'{model}_correct' for model in model_names]
                all_agree = self.comparison_data[all_correct_cols].apply(
                    lambda row: len(set(row)) == 1, axis=1
                ).mean()
                agreement_stats['all_models_agree'] = all_agree * 100
        
        return agreement_stats
    
    def analyze_difficulty(self):
        """Analyze question difficulty based on how many models got it right."""
        if self.comparison_data is None:
            return None
        
        model_names = list(self.models.keys())
        correct_cols = [f'{model}_correct' for model in model_names]
        
        if all(col in self.comparison_data.columns for col in correct_cols):
            self.comparison_data['models_correct'] = self.comparison_data[correct_cols].sum(axis=1)
            
            difficulty_analysis = self.comparison_data['models_correct'].value_counts().sort_index()
            difficulty_percentages = (difficulty_analysis / len(self.comparison_data) * 100).round(2)
            
            return {
                'distribution': difficulty_analysis.to_dict(),
                'percentages': difficulty_percentages.to_dict()
            }
        
        return None
    
    def find_unique_errors(self):
        """Find questions where only one model made an error."""
        if self.comparison_data is None:
            return None
        
        model_names = list(self.models.keys())
        correct_cols = [f'{model}_correct' for model in model_names]
        
        unique_errors = {}
        
        for model in model_names:
            model_col = f'{model}_correct'
            other_cols = [col for col in correct_cols if col != model_col]
            
            # Questions where this model is wrong but others are right
            if len(other_cols) > 0:
                condition = (self.comparison_data[model_col] == False)
                for other_col in other_cols:
                    condition = condition & (self.comparison_data[other_col] == True)
                
                unique_error_questions = self.comparison_data[condition]
                unique_errors[model] = len(unique_error_questions)
        
        return unique_errors
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Calculate metrics for plotting
        metrics = self.calculate_basic_metrics()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dental AI Models Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        models = list(metrics.keys())
        accuracies = [metrics[model]['accuracy_percent'] for model in models]
        
        bars = axes[0, 0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Correct vs Incorrect Questions
        correct_counts = [metrics[model]['correct_answers'] for model in models]
        incorrect_counts = [metrics[model]['incorrect_answers'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, correct_counts, width, label='Correct', color='#2ECC71')
        axes[0, 1].bar(x + width/2, incorrect_counts, width, label='Incorrect', color='#E74C3C')
        
        axes[0, 1].set_title('Correct vs Incorrect Answers', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Questions')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        
        # 3. Question Difficulty Analysis (if comparison data available)
        if self.comparison_data is not None:
            difficulty = self.analyze_difficulty()
            if difficulty:
                models_correct = list(difficulty['distribution'].keys())
                question_counts = list(difficulty['distribution'].values())
                
                colors = ['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71']
                bars = axes[1, 0].bar(models_correct, question_counts, color=colors[:len(models_correct)])
                axes[1, 0].set_title('Question Difficulty Distribution', fontweight='bold')
                axes[1, 0].set_xlabel('Number of Models That Got It Right')
                axes[1, 0].set_ylabel('Number of Questions')
                
                # Add percentage labels
                total_questions = sum(question_counts)
                for bar, count in zip(bars, question_counts):
                    percentage = count / total_questions * 100
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                   f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 4. Model Agreement Analysis
        if self.comparison_data is not None:
            agreement_stats = self.analyze_agreement()
            if agreement_stats:
                agreement_labels = list(agreement_stats.keys())
                agreement_values = list(agreement_stats.values())
                
                bars = axes[1, 1].bar(range(len(agreement_labels)), agreement_values, 
                                     color=['#9B59B6', '#3498DB', '#1ABC9C'])
                axes[1, 1].set_title('Model Agreement Analysis', fontweight='bold')
                axes[1, 1].set_ylabel('Agreement (%)')
                axes[1, 1].set_xticks(range(len(agreement_labels)))
                axes[1, 1].set_xticklabels(agreement_labels, rotation=45, ha='right')
                axes[1, 1].set_ylim(0, 100)
                
                # Add value labels
                for bar, val in zip(bars, agreement_values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                                   f'{val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plt.savefig('dental_analysis_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'dental_analysis_results.png'")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*80)
        print("DENTAL AI MODELS PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        # Basic metrics
        metrics = self.calculate_basic_metrics()
        print("\n1. BASIC PERFORMANCE METRICS")
        print("-" * 40)
        for model, metric in metrics.items():
            print(f"\n{model}:")
            print(f"  Total Questions: {metric['total_questions']:,}")
            print(f"  Correct Answers: {metric['correct_answers']:,}")
            print(f"  Incorrect Answers: {metric['incorrect_answers']:,}")
            print(f"  Accuracy: {metric['accuracy_percent']:.2f}%")
        
        # Model ranking
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['accuracy_percent'], reverse=True)
        print(f"\n2. MODEL RANKING (by Accuracy)")
        print("-" * 40)
        for rank, (model, metric) in enumerate(sorted_models, 1):
            print(f"{rank}. {model}: {metric['accuracy_percent']:.2f}%")
        
        # Comparison analysis
        if self.comparison_data is not None:
            print(f"\n3. CROSS-MODEL ANALYSIS")
            print("-" * 40)
            print(f"Questions common to all models: {len(self.comparison_data):,}")
            
            # Agreement analysis
            agreement_stats = self.analyze_agreement()
            if agreement_stats:
                print(f"\nModel Agreement:")
                for pair, agreement in agreement_stats.items():
                    print(f"  {pair}: {agreement:.2f}%")
            
            # Difficulty analysis
            difficulty = self.analyze_difficulty()
            if difficulty:
                print(f"\nQuestion Difficulty Distribution:")
                for num_correct, count in difficulty['distribution'].items():
                    percentage = difficulty['percentages'][num_correct]
                    print(f"  {num_correct} models correct: {count} questions ({percentage}%)")
            
            # Unique errors
            unique_errors = self.find_unique_errors()
            if unique_errors:
                print(f"\nUnique Errors (where only one model failed):")
                for model, count in unique_errors.items():
                    print(f"  {model}: {count} unique errors")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting Dental Results Analysis...")
        print("="*50)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Create comparison data if multiple models available
        if len(self.models) > 1:
            self.create_comparison_dataframe()
        
        # Generate visualizations
        self.visualize_results()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\nAnalysis complete!")


def main():
    """Main function to run the analysis."""
    analyzer = DentalResultsAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
