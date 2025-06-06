#!/usr/bin/env python3
"""
Run dental benchmarks for multiple models
"""
import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_gpt4_nano_benchmark(args):
    """Run GPT-4-1-nano benchmark"""
    logger.info("Starting GPT-4-1-nano benchmark...")
    
    cmd = [
        sys.executable, 
        "gpt-4-1-nano/gpt4_nano_benchmark.py"
    ]
    
    if args.openai_key:
        cmd.extend(["--api-key", args.openai_key])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"gpt4_nano_results_{timestamp}.json")
        cmd.extend(["--output", output_path])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("GPT-4-1-nano benchmark completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"GPT-4-1-nano benchmark failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def run_o3_mini_benchmark(args):
    """Run O3-mini benchmark"""
    logger.info("Starting O3-mini benchmark...")
    
    cmd = [
        sys.executable, 
        "o3-mini/o3_mini_benchmark.py"
    ]
    
    if args.openai_key:
        cmd.extend(["--api-key", args.openai_key])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.mock_o3:
        cmd.append("--mock")
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"o3_mini_results_{timestamp}.json")
        cmd.extend(["--output", output_path])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("O3-mini benchmark completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"O3-mini benchmark failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run dental benchmarks for multiple OpenAI models')
    
    # Model selection
    parser.add_argument('--models', nargs='+', choices=['gpt4-nano', 'o3-mini', 'all'],
                      default=['all'], help='Which models to benchmark')
    
    # API key (unified for both OpenAI models)
    parser.add_argument('--openai-key', help='OpenAI API key for both models (or set OPENAI_API_KEY env var)')
    
    # General options
    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--mock-o3', action='store_true', help='Use mock responses for O3-mini')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if API key is available
    api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not api_key and not args.mock_o3:
        logger.warning("No OpenAI API key provided. O3-mini will use mock responses.")
        logger.warning("Set OPENAI_API_KEY environment variable or use --openai-key argument.")
        logger.warning("Use --mock-o3 flag to explicitly enable mock mode.")
    
    # Determine which models to run
    if 'all' in args.models:
        models_to_run = ['gpt4-nano', 'o3-mini']
    else:
        models_to_run = args.models
    
    results = {}
    
    # Run benchmarks
    if 'gpt4-nano' in models_to_run:
        results['gpt4-nano'] = run_gpt4_nano_benchmark(args)
    
    if 'o3-mini' in models_to_run:
        results['o3-mini'] = run_o3_mini_benchmark(args)
    
    # Print summary
    print("\n" + "="*60)
    print("DENTAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"OpenAI API Key: {'✅ Provided' if api_key else '❌ Not provided'}")
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    print(f"\nResults saved to: {args.output_dir}")
    
    # Exit with error code if any benchmark failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main() 