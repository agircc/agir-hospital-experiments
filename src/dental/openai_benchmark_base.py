"""
OpenAI benchmark base class with checkpoint support
"""
import os
import sys
import json
import openai
from typing import Dict, Any
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import base class from same directory
from benchmark_base import DentalBenchmark

logger = logging.getLogger(__name__)

class OpenAIBenchmark(DentalBenchmark):
    """Base class for OpenAI model benchmarking with checkpoint support"""
    
    def __init__(self, model_name: str, model_id: str, api_key: str = None, 
                 data_path: str = None):
        # Set default data path if not provided
        if data_path is None:
            # Find project root by looking for .git directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != os.path.dirname(project_root):  # Not at filesystem root
                if os.path.exists(os.path.join(project_root, '.git')) or os.path.exists(os.path.join(project_root, 'Makefile')):
                    break
                project_root = os.path.dirname(project_root)
            data_path = os.path.join(project_root, "datasets_by_subject", "dental_valid.jsonl")
        
        super().__init__(model_name, data_path)
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configuration
        self.model_id = model_id
        self.max_tokens = 5  # Only need one letter (A, B, C, or D)
        self.temperature = 0.1  # Low temperature for consistent medical answers
        
        # Checkpoint configuration - use project root
        # Find project root by looking for .git directory or Makefile
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while project_root != os.path.dirname(project_root):  # Not at filesystem root
            if os.path.exists(os.path.join(project_root, '.git')) or os.path.exists(os.path.join(project_root, 'Makefile')):
                break
            project_root = os.path.dirname(project_root)
        

    def query_model(self, prompt: str) -> str:
        """Query OpenAI model"""
        try:
            # Prepare API call parameters
            params = {
                'model': self.model_id,
                'messages': [
                    {
                        "role": "system", 
                        "content": "You are a medical expert. Answer multiple choice questions with only the letter (A, B, C, or D). Do not provide explanations."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            }
            
            # Use different parameters based on model
            if 'o3' in self.model_id.lower():
                # O3 models don't support additional parameters
                # Keep only the basic required parameters
                pass
            else:
                # Other models use max_tokens and temperature
                params['max_tokens'] = self.max_tokens
                params['temperature'] = self.temperature
            
            response = self.client.chat.completions.create(**params)
            
            return response.choices[0].message.content.strip()
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error querying {self.model_name}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error querying {self.model_name}: {e}")
            raise e
    
    def get_completed_count(self) -> int:
        """Get count of completed questions from CSV file"""
        if not os.path.exists(self.csv_path):
            return 0
        
        try:
            import csv
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                # Count rows excluding header
                return max(0, len(rows) - 1)  # -1 for header
        except Exception as e:
            logger.error(f"Error reading existing CSV: {e}")
            return 0
    
    def run_benchmark(self, limit: int = None) -> Dict[str, Any]:
        """Run the benchmark, continuing from existing progress"""
        import time
        
        logger.info(f"Starting {self.model_name} benchmark on dental test set")
        
        # Load full test data 
        if not hasattr(self, 'questions') or not self.questions:
            self.load_test_data()
        
        total_questions = len(self.questions)
        completed_count = self.get_completed_count()
        
        logger.info(f"Dataset has {total_questions} total questions")
        logger.info(f"Already completed {completed_count} questions")
        
        # Check if already completed all
        if completed_count >= total_questions:
            logger.info(f"✅ All {total_questions} questions already completed!")
            logger.info(f"CSV results are saved at: {self.csv_path}")
            return {
                'model_name': self.model_name,
                'model_id': self.model_id,
                'total_questions': total_questions,
                'completed_questions': completed_count,
                'new_questions': 0,
                'correct_answers': 0,
                'accuracy': 0,
                'duration_seconds': 0,
                'timestamp': datetime.now().isoformat(),
                'results': [],
                'status': 'already_completed'
            }
        
        # Determine how many questions to run this time
        remaining_questions = total_questions - completed_count
        if limit:
            questions_to_run = min(limit, remaining_questions)
        else:
            questions_to_run = remaining_questions
            
        start_index = completed_count
        end_index = start_index + questions_to_run
        
        logger.info(f"Will process questions {start_index + 1} to {end_index} ({questions_to_run} questions)")
        
        # CSV is already set up in constructor - it will append if file exists, create if not
        
        correct_answers = 0
        start_time = time.time()
        self.results = []
        
        for i in range(start_index, end_index):
            question_data = self.questions[i]
            logger.info(f"Processing question {i+1}/{total_questions} (#{i-start_index+1} of this run)")
            
            # Format question
            prompt = self.format_question(question_data)
            
            # Query model
            try:
                response = self.query_model(prompt)
                predicted_answer = self.extract_answer_choice(response)
                is_correct = self.evaluate_answer(predicted_answer, question_data['cop'])
                
                if is_correct:
                    correct_answers += 1
                
                # Store result
                result = {
                    'question_id': question_data['id'],
                    'question': question_data['question'],
                    'correct_option': self.get_correct_option_letter(question_data['cop']),
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response,
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
                # Write result to CSV immediately
                self.write_result_to_csv(result)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                # Store error result
                result = {
                    'question_id': question_data['id'],
                    'question': question_data['question'],
                    'correct_option': self.get_correct_option_letter(question_data['cop']),
                    'predicted_answer': 'ERROR',
                    'is_correct': False,
                    'response': f"Error: {e}",
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
                # Write error result to CSV immediately
                self.write_result_to_csv(result)
        
        end_time = time.time()
        duration = end_time - start_time
        accuracy = correct_answers / questions_to_run if questions_to_run > 0 else 0
        
        # Final status
        final_completed = completed_count + questions_to_run
        final_accuracy_str = f"({accuracy:.2%} this run)" if questions_to_run > 0 else ""
        
        logger.info(f"Completed {questions_to_run} questions in {duration:.2f} seconds")
        logger.info(f"This run: {correct_answers}/{questions_to_run} correct {final_accuracy_str}")
        logger.info(f"Total progress: {final_completed}/{total_questions}")
        
        # Compile final results
        benchmark_results = {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'total_questions': total_questions,
            'completed_questions': final_completed,
            'new_questions': questions_to_run,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        return benchmark_results
    
 