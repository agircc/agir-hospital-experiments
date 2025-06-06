"""
GPT-4-1-nano benchmark implementation for dental subject
"""
import os
import sys
import openai
from typing import Dict, Any
import logging

# Add parent directory to path to import base class
sys.path.append('..')
from benchmark_base import DentalBenchmark

logger = logging.getLogger(__name__)

class GPT4NanoBenchmark(DentalBenchmark):
    """GPT-4-1-nano implementation for dental benchmarking"""
    
    def __init__(self, api_key: str = None, data_path: str = "../../../datasets_by_subject/dental_test.jsonl"):
        super().__init__("gpt-4-1-nano", data_path)
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configuration
        self.model_id = "gpt-4o-mini"  # Using available model as placeholder for gpt-4-1-nano
        self.max_tokens = 500
        self.temperature = 0.1  # Low temperature for consistent medical answers
        
    def query_model(self, prompt: str) -> str:
        """Query GPT-4-1-nano model"""
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
            
        except Exception as e:
            logger.error(f"Error querying GPT-4-1-nano: {e}")
            raise e

def main():
    """Main function to run GPT-4-1-nano benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GPT-4-1-nano benchmark on dental test set')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--data-path', default='../../../datasets_by_subject/dental_test.jsonl',
                      help='Path to dental test data')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = GPT4NanoBenchmark(api_key=args.api_key, data_path=args.data_path)
        
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
        print("GPT-4-1-nano Dental Benchmark Results")
        print("="*50)
        print(f"Model: {results['model_name']}")
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