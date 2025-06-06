"""
Base class for dental subject benchmarking
"""
import json
import os
import time
import csv
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DentalBenchmark(ABC):
    """Base class for dental subject benchmarking"""
    
    def __init__(self, model_name: str, data_path: str = "../../datasets_by_subject/dental_valid.jsonl"):
        self.model_name = model_name
        self.data_path = data_path
        self.questions = []
        self.results = []
        
        # Setup CSV output path
        self.csv_path = self._setup_csv_output()
        
    def _setup_csv_output(self) -> str:
        """Setup CSV output file path and create directory if needed"""
        # Find project root by looking for .git directory or Makefile
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while project_root != os.path.dirname(project_root):  # Not at filesystem root
            if os.path.exists(os.path.join(project_root, '.git')) or os.path.exists(os.path.join(project_root, 'Makefile')):
                break
            project_root = os.path.dirname(project_root)
        
        # Create results/dental directory if it doesn't exist
        results_dir = os.path.join(project_root, "results", "dental")
        os.makedirs(results_dir, exist_ok=True)
        
        csv_path = os.path.join(results_dir, f"{self.model_name}_dental_results.csv")
        
        # Initialize CSV file with headers
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'question_id', 'question', 'correct_option', 'predicted_answer', 
                'is_correct', 'response', 'topic', 'subject'
            ])
        
        logger.info(f"CSV output initialized: {csv_path}")
        return csv_path
    
    def write_result_to_csv(self, result: Dict[str, Any]):
        """Write a single result to CSV file"""
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                result['question_id'],
                result['question'][:200] + '...' if len(result['question']) > 200 else result['question'],
                result['correct_option'],
                result['predicted_answer'],
                result['is_correct'],
                result['response'][:100] + '...' if len(result['response']) > 100 else result['response'],
                result['topic'],
                result['subject']
            ])
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load dental test data from JSONL file"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Test data not found at {self.data_path}")
            
        questions = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        logger.info(f"Loaded {len(questions)} dental test questions")
        self.questions = questions
        return questions
    
    def format_question(self, question_data: Dict[str, Any]) -> str:
        """Format question for model input"""
        question = question_data['question']
        options = [
            f"A) {question_data['opa']}",
            f"B) {question_data['opb']}",
            f"C) {question_data['opc']}",
            f"D) {question_data['opd']}"
        ]
        
        formatted = f"""Medical Question (Dental):
{question}

Options:
{chr(10).join(options)}

Please select the correct answer and respond with only the letter (A, B, C, or D)."""
        
        return formatted
    
    def extract_answer_choice(self, response: str) -> str:
        """Extract the answer choice (A, B, C, D) from model response"""
        response_clean = response.strip().upper()
        
        # If response is already just a single letter
        if response_clean in ['A', 'B', 'C', 'D']:
            return response_clean
        
        # Look for first occurrence of A, B, C, or D in the response
        for char in ['A', 'B', 'C', 'D']:
            if char in response_clean:
                return char
                
        return "UNKNOWN"
    
    def evaluate_answer(self, predicted: str, correct_option: int) -> bool:
        """Evaluate if predicted answer matches correct option"""
        # Convert numeric option to letter (0-based indexing)
        option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_letter = option_map.get(correct_option, '')
        return predicted == correct_letter
    
    def get_correct_option_letter(self, correct_option: int) -> str:
        """Convert numeric correct option to letter format"""
        option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', -1: 'UNKNOWN'}
        return option_map.get(correct_option, 'UNKNOWN')
    
    @abstractmethod
    def query_model(self, prompt: str) -> str:
        """Query the specific model - to be implemented by subclasses"""
        pass
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark"""
        logger.info(f"Starting {self.model_name} benchmark on dental test set")
        
        # Load test data
        self.load_test_data()
        
        correct_answers = 0
        total_questions = len(self.questions)
        start_time = time.time()
        
        for i, question_data in enumerate(self.questions):
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
                    'correct_option': self.get_correct_option_letter(question_data['cop']),
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response,
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
                # Write result to CSV
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
                
                # Write error result to CSV
                self.write_result_to_csv(result)
        
        end_time = time.time()
        duration = end_time - start_time
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        
        # Compile final results
        benchmark_results = {
            'model_name': self.model_name,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        logger.info(f"Benchmark completed: {correct_answers}/{total_questions} correct ({accuracy:.2%})")
        logger.info(f"Duration: {duration:.2f} seconds")
        
        return benchmark_results
    
 