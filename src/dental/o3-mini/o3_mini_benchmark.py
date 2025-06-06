"""
O3-mini benchmark implementation for dental subject
"""
import os
import sys
import openai
from typing import Dict, Any
import logging
import time

# Add parent directory to path to import base class
sys.path.append('..')
from benchmark_base import DentalBenchmark

logger = logging.getLogger(__name__)

class O3MiniBenchmark(DentalBenchmark):
    """O3-mini implementation for dental benchmarking"""
    
    def __init__(self, api_key: str = None, data_path: str = "../../../datasets_by_subject/dental_test.jsonl"):
        super().__init__("o3-mini", data_path)
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Using mock responses for demonstration.")
            self.use_mock = True
        else:
            self.use_mock = False
            self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configuration
        self.model_id = "o3-mini"  # Official OpenAI o3-mini model
        self.max_tokens = 500
        self.temperature = 0.1  # Low temperature for consistent medical answers
        
    def query_model(self, prompt: str) -> str:
        """Query O3-mini model via OpenAI API"""
        if self.use_mock:
            return self._mock_response(prompt)
        
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
            logger.error(f"OpenAI API error querying O3-mini: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error querying O3-mini: {e}")
            raise e
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for demonstration purposes"""
        # Add a small delay to simulate API call
        time.sleep(0.5)
        
        # Extract question to generate a reasonable mock answer
        if "Options:" in prompt:
            # Simple heuristic: choose option A for demonstration
            mock_responses = [
                "Looking at this dental question, I need to consider the clinical presentation and pathophysiology. Based on the symptoms and clinical findings described, the correct answer is A. This condition is characterized by the specific dental manifestations mentioned in the question.",
                "As a dental expert, I analyze this case systematically. The key indicators point to option B. This diagnosis is consistent with the clinical presentation and follows established dental medical guidelines.",
                "Examining the dental pathology described, the most appropriate answer is C. This condition typically presents with the symptoms outlined and requires the treatment approach mentioned in this option.",
                "From a dental medicine perspective, the correct answer is D. The clinical features and diagnostic criteria clearly align with this condition, making it the most appropriate choice."
            ]
            
            # Rotate through responses based on prompt hash for consistency
            response_idx = hash(prompt) % len(mock_responses)
            return mock_responses[response_idx]
        
        return "Based on my analysis as a dental expert, the correct answer is A. This represents the most clinically appropriate choice given the presented case."

def main():
    """Main function to run O3-mini benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run O3-mini benchmark on dental test set')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--data-path', default='../../../datasets_by_subject/dental_test.jsonl',
                      help='Path to dental test data')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    parser.add_argument('--mock', action='store_true', help='Force use of mock responses')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = O3MiniBenchmark(
            api_key=args.api_key,
            data_path=args.data_path
        )
        
        if args.mock:
            benchmark.use_mock = True
            logger.info("Using mock responses as requested")
        
        # Limit questions if specified (for testing)
        if args.limit:
            benchmark.load_test_data()
            benchmark.questions = benchmark.questions[:args.limit]
            logger.info(f"Limited to {args.limit} questions for testing")
        
        # Run benchmark
        results = benchmark.run_benchmark()
        
        # Save results
        output_path = benchmark.save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("O3-mini Dental Benchmark Results")
        print("="*50)
        print(f"Model: {results['model_name']}")
        if benchmark.use_mock:
            print("⚠️  Using mock responses (O3-mini API not available)")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 