"""
Base class for dental subject benchmarking
"""
import json
import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DentalBenchmark(ABC):
    """Base class for dental subject benchmarking"""
    
    def __init__(self, model_name: str, data_path: str = "../../datasets_by_subject/dental_test.jsonl"):
        self.model_name = model_name
        self.data_path = data_path
        self.questions = []
        self.results = []
        
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

Please select the correct answer (A, B, C, or D) and briefly explain your reasoning."""
        
        return formatted
    
    def extract_answer_choice(self, response: str) -> str:
        """Extract the answer choice (A, B, C, D) from model response"""
        response_upper = response.upper()
        
        # Look for explicit answer patterns
        patterns = [
            "ANSWER: ",
            "ANSWER IS ",
            "CORRECT ANSWER: ",
            "THE ANSWER IS ",
            "I CHOOSE ",
            "OPTION "
        ]
        
        for pattern in patterns:
            if pattern in response_upper:
                idx = response_upper.find(pattern) + len(pattern)
                next_char = response_upper[idx:idx+1]
                if next_char in ['A', 'B', 'C', 'D']:
                    return next_char
        
        # Look for first occurrence of A, B, C, or D
        for char in ['A', 'B', 'C', 'D']:
            if char in response_upper:
                return char
                
        return "UNKNOWN"
    
    def evaluate_answer(self, predicted: str, correct_option: int) -> bool:
        """Evaluate if predicted answer matches correct option"""
        option_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        correct_letter = option_map.get(correct_option, '')
        return predicted == correct_letter
    
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
                    'correct_option': question_data['cop'],
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'response': response,
                    'topic': question_data.get('topic_name', ''),
                    'subject': question_data.get('subject_name', 'Dental')
                }
                self.results.append(result)
                
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
    
    def save_results(self, results: Dict[str, Any], output_path: str = None) -> str:
        """Save benchmark results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{self.model_name}_dental_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path 