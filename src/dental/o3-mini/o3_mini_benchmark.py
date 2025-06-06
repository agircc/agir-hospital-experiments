"""
O3-mini benchmark implementation for dental subject
"""
import os
import sys
import logging

# Add parent directory to path to import base class
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai_benchmark_base import OpenAIBenchmark

logger = logging.getLogger(__name__)

class O3MiniBenchmark(OpenAIBenchmark):
    """O3-mini implementation for dental benchmarking"""
    
    def __init__(self, api_key: str = None, data_path: str = None):
        # Initialize with correct model name and ID
        super().__init__("o3-mini", "o3-mini", api_key, data_path)

def main():
    """Main function to run O3-mini benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run O3-mini benchmark on dental test set')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--data-path', 
                      help='Path to dental test data (default: auto-detect project root)')

    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    parser.add_argument('--save-frequency', type=int, default=5, 
                      help='Save checkpoint every N questions (default: 5)')
    parser.add_argument('--clear-checkpoint', action='store_true',
                      help='Clear existing checkpoint and start fresh')

    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = O3MiniBenchmark(api_key=args.api_key, data_path=args.data_path)
        
        # Clear checkpoint if requested
        if args.clear_checkpoint:
            benchmark.clear_checkpoint()
            logger.info("Cleared existing checkpoint")
        
        # Limit questions if specified (for testing)
        if args.limit:
            benchmark.load_test_data()
            benchmark.questions = benchmark.questions[:args.limit]
            logger.info(f"Limited to {args.limit} questions for testing")
        
        # Run benchmark with checkpoint support
        results = benchmark.run_benchmark(save_frequency=args.save_frequency)
        
        # Print summary
        print("\n" + "="*50)
        print("O3-mini Dental Benchmark Results")
        print("="*50)
        print(f"Model: {results['model_name']} ({results['model_id']})")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"CSV results saved to: {benchmark.csv_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 