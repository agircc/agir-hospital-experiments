"""
GPT-4.1-nano benchmark implementation for dental subject
"""
import os
import sys
import logging

# Add parent directory to path to import base class
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai_benchmark_base import OpenAIBenchmark

logger = logging.getLogger(__name__)

class GPT41NanoBenchmark(OpenAIBenchmark):
    """GPT-4.1-nano implementation for dental benchmarking"""
    
    def __init__(self, api_key: str = None, data_path: str = None):
        # Initialize with correct model name and ID
        super().__init__("gpt-4.1-nano", "gpt-4.1-nano", api_key, data_path)

def main():
    """Main function to run GPT-4.1-nano benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GPT-4.1-nano benchmark on dental test set')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--data-path', 
                      help='Path to dental test data (default: auto-detect project root)')

    parser.add_argument('--limit', type=int, help='Number of additional questions to process (default: all remaining)')
    parser.add_argument('--clear-results', action='store_true',
                      help='Clear existing results and start fresh')

    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = GPT41NanoBenchmark(api_key=args.api_key, data_path=args.data_path)
        
        # Clear results if requested
        if args.clear_results:
            import os
            if os.path.exists(benchmark.csv_path):
                os.remove(benchmark.csv_path)
                print("✅ Cleared existing results")
                print(f"Removed: {benchmark.csv_path}")
                return
            else:
                print("ℹ️  No existing results to clear")
                return
        
        # Run benchmark 
        results = benchmark.run_benchmark(limit=args.limit)
        
        # Print summary
        print("\n" + "="*50)
        if results.get('status') == 'already_completed':
            print("GPT-4.1-nano Dental Benchmark - Already Completed")
            print("="*50)
            print(f"✅ All {results['total_questions']} questions have been processed.")
            print(f"CSV results available at: {benchmark.csv_path}")
        else:
            print("GPT-4.1-nano Dental Benchmark Results")
            print("="*50)
            print(f"Model: {results['model_name']} ({results['model_id']})")
            print(f"Dataset Total: {results['total_questions']} questions")
            print(f"Total Completed: {results['completed_questions']}/{results['total_questions']}")
            print(f"This Run: {results['new_questions']} questions processed")
            print(f"This Run Accuracy: {results['correct_answers']}/{results['new_questions']} ({results['accuracy']:.2%})")
            print(f"Duration: {results['duration_seconds']:.2f} seconds")
            print(f"CSV results saved to: {benchmark.csv_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 