"""
GPT-4.1-nano benchmark implementation for dental subject
"""
import os
import sys
import logging

# Add parent directory to path to import base class
sys.path.append('..')
from openai_benchmark_base import OpenAIBenchmark

logger = logging.getLogger(__name__)

class GPT41NanoBenchmark(OpenAIBenchmark):
    """GPT-4.1-nano implementation for dental benchmarking"""
    
    def __init__(self, api_key: str = None, data_path: str = "../../../datasets_by_subject/dental_test.jsonl"):
        # Initialize with correct model name and ID
        super().__init__("gpt-4.1-nano", "gpt-4.1-nano", api_key, data_path)

def main():
    """Main function to run GPT-4.1-nano benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GPT-4.1-nano benchmark on dental test set')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--data-path', default='../../../datasets_by_subject/dental_test.jsonl',
                      help='Path to dental test data')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    parser.add_argument('--save-frequency', type=int, default=5, 
                      help='Save checkpoint every N questions (default: 5)')
    parser.add_argument('--clear-checkpoint', action='store_true',
                      help='Clear existing checkpoint and start fresh')
    parser.add_argument('--export-csv', action='store_true',
                      help='Also export results to CSV format')
    
    args = parser.parse_args()
    
    try:
        # Initialize benchmark
        benchmark = GPT41NanoBenchmark(api_key=args.api_key, data_path=args.data_path)
        
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
        
        # Save results
        output_path = benchmark.save_results(results, args.output)
        
        # Export to CSV if requested
        csv_path = None
        if args.export_csv:
            csv_base = args.output.replace('.json', '.csv') if args.output else None
            csv_path = benchmark.save_results_csv(results, csv_base)
        
        # Print summary
        print("\n" + "="*50)
        print("GPT-4.1-nano Dental Benchmark Results")
        print("="*50)
        print(f"Model: {results['model_name']} ({results['model_id']})")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Results saved to: {output_path}")
        if csv_path:
            print(f"CSV exported to: {csv_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 