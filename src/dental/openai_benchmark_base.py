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
                 data_path: str = None,
                 checkpoint_dir: str = "checkpoints"):
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
        self.max_tokens = 500
        self.temperature = 0.1  # Low temperature for consistent medical answers
        
        # Checkpoint configuration
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{model_name}_checkpoint.json")
        self.progress_file = os.path.join(self.checkpoint_dir, f"{model_name}_progress.json")
        
    def query_model(self, prompt: str) -> str:
        """Query OpenAI model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a medical expert specializing in dental medicine. Answer multiple choice questions accurately and provide clear reasoning."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error querying {self.model_name}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error querying {self.model_name}: {e}")
            raise e
    
    def save_checkpoint(self, current_index: int, results_so_far: list):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'model_name': self.model_name,
            'current_index': current_index,
            'total_questions': len(self.questions),
            'results_so_far': results_so_far,
            'timestamp': datetime.now().isoformat(),
            'data_path': self.data_path
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Save progress summary
        progress_data = {
            'completed': current_index,
            'total': len(self.questions),
            'percentage': (current_index / len(self.questions)) * 100 if self.questions else 0,
            'correct_so_far': sum(1 for r in results_so_far if r.get('is_correct', False)),
            'accuracy_so_far': sum(1 for r in results_so_far if r.get('is_correct', False)) / len(results_so_far) if results_so_far else 0
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Checkpoint saved: {current_index}/{len(self.questions)} questions completed")
    
    def load_checkpoint(self) -> tuple:
        """Load checkpoint if exists. Returns (start_index, existing_results)"""
        if not os.path.exists(self.checkpoint_file):
            return 0, []
        
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Verify checkpoint is for the same data
            if checkpoint_data.get('data_path') != self.data_path:
                logger.warning("Checkpoint data path mismatch. Starting from beginning.")
                return 0, []
            
            if checkpoint_data.get('total_questions') != len(self.questions):
                logger.warning("Question count mismatch. Starting from beginning.")
                return 0, []
            
            start_index = checkpoint_data.get('current_index', 0)
            existing_results = checkpoint_data.get('results_so_far', [])
            
            logger.info(f"Resuming from checkpoint: {start_index}/{len(self.questions)} questions completed")
            return start_index, existing_results
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0, []
    
    def clear_checkpoint(self):
        """Clear checkpoint files"""
        for file_path in [self.checkpoint_file, self.progress_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed checkpoint file: {file_path}")
    
    def run_benchmark(self, save_frequency: int = 5) -> Dict[str, Any]:
        """Run the complete benchmark with checkpoint support"""
        import time
        
        logger.info(f"Starting {self.model_name} benchmark on dental test set")
        
        # Load test data if not already loaded
        if not hasattr(self, 'questions') or not self.questions:
            self.load_test_data()
        
        # Load checkpoint
        start_index, existing_results = self.load_checkpoint()
        
        correct_answers = sum(1 for r in existing_results if r.get('is_correct', False))
        total_questions = len(self.questions)
        start_time = time.time()
        
        # Update results with existing ones
        self.results = existing_results.copy()
        
        if start_index > 0:
            logger.info(f"Resuming from question {start_index + 1}/{total_questions}")
            logger.info(f"Current accuracy: {correct_answers}/{start_index} ({correct_answers/start_index:.2%})" if start_index > 0 else "")
        
        for i in range(start_index, total_questions):
            question_data = self.questions[i]
            logger.info(f"Processing question {i+1}/{total_questions}")
            
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
                    'correct_option': question_data['cop'],
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response,
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
                # Save checkpoint periodically
                if (i + 1) % save_frequency == 0:
                    self.save_checkpoint(i + 1, self.results)
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                # Store error result
                result = {
                    'question_id': question_data['id'],
                    'question': question_data['question'],
                    'correct_option': question_data['cop'],
                    'predicted_answer': 'ERROR',
                    'is_correct': False,
                    'response': f"Error: {e}",
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
                # Save checkpoint on error too
                self.save_checkpoint(i + 1, self.results)
        
        end_time = time.time()
        duration = end_time - start_time
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Compile final results
        benchmark_results = {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        logger.info(f"Benchmark completed: {correct_answers}/{total_questions} correct ({accuracy:.2%})")
        logger.info(f"Duration: {duration:.2f} seconds")
        
        # Clear checkpoint after successful completion
        self.clear_checkpoint()
        
        return benchmark_results
    
    def save_results_csv(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save benchmark results to CSV file for data analysis"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.model_name}_dental_results_{timestamp}.csv"
        
        # Flatten results for CSV format
        flattened_results = []
        for result in results['results']:
            flat_result = {
                'model_name': results['model_name'],
                'model_id': results['model_id'],
                'question_id': result['question_id'],
                'question': result['question'][:200] + '...' if len(result['question']) > 200 else result['question'],  # Truncate long questions
                'correct_option': result['correct_option'],
                'predicted_answer': result['predicted_answer'],
                'is_correct': result['is_correct'],
                'topic': result['topic'],
                'subject': result['subject'],
                'response_length': len(result['response'])
            }
            flattened_results.append(flat_result)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"CSV results saved to {output_path}")
        return output_path 