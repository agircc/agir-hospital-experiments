#!/usr/bin/env python3
"""
Run dental benchmarks for multiple models with checkpoint support
"""
import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_gpt41_nano_benchmark(args):
    """Run GPT-4.1-nano benchmark"""
    logger.info("Starting GPT-4.1-nano benchmark...")
    
    cmd = [
        sys.executable, 
        "gpt-4-1-nano/gpt4_nano_benchmark.py"
    ]
    
    if args.openai_key:
        cmd.extend(["--api-key", args.openai_key])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.save_frequency:
        cmd.extend(["--save-frequency", str(args.save_frequency)])
    if args.clear_checkpoints:
        cmd.append("--clear-checkpoint")
    if args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"gpt41_nano_results_{timestamp}.json")
        cmd.extend(["--output", output_path])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("GPT-4.1-nano benchmark completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"GPT-4.1-nano benchmark failed: {e}")
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
    if args.save_frequency:
        cmd.extend(["--save-frequency", str(args.save_frequency)])
    if args.clear_checkpoints:
        cmd.append("--clear-checkpoint")
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

def check_checkpoints():
    """Check for existing checkpoints"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith("_checkpoint.json"):
            model_name = file.replace("_checkpoint.json", "")
            checkpoints.append(model_name)
    
    return checkpoints

def main():
    parser = argparse.ArgumentParser(description='Run dental benchmarks for multiple OpenAI models with checkpoint support')
    
    # Model selection
    parser.add_argument('--models', nargs='+', choices=['gpt4.1-nano', 'o3-mini', 'all'],
                      default=['all'], help='Which models to benchmark')
    
    # API key (unified for both OpenAI models)
    parser.add_argument('--openai-key', help='OpenAI API key for both models (or set OPENAI_API_KEY env var)')
    
    # General options
    parser.add_argument('--limit', type=int, help='Limit number of questions for testing')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--save-frequency', type=int, default=5,
                      help='Save checkpoint every N questions (default: 5)')
    parser.add_argument('--clear-checkpoints', action='store_true',
                      help='Clear existing checkpoints and start fresh')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check existing checkpoints
    existing_checkpoints = check_checkpoints()
    if existing_checkpoints and not args.clear_checkpoints:
        logger.info(f"Found existing checkpoints: {existing_checkpoints}")
        logger.info("Use --clear-checkpoints to start fresh or continue from where you left off")
    elif existing_checkpoints and args.clear_checkpoints:
        logger.info(f"Will clear existing checkpoints: {existing_checkpoints}")
    
    # Check if API key is available
    api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("No OpenAI API key provided.")
        logger.error("Set OPENAI_API_KEY environment variable or use --openai-key argument.")
        sys.exit(1)
    
    # Determine which models to run
    if 'all' in args.models:
        models_to_run = ['gpt4.1-nano', 'o3-mini']
    else:
        models_to_run = args.models
    
    results = {}
    
    # Run benchmarks
    if 'gpt4.1-nano' in models_to_run:
        results['gpt4.1-nano'] = run_gpt41_nano_benchmark(args)
    
    if 'o3-mini' in models_to_run:
        results['o3-mini'] = run_o3_mini_benchmark(args)
    
    # Print summary
    print("\n" + "="*60)
    print("DENTAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"OpenAI API Key: {'✅ Provided' if api_key else '❌ Not provided'}")
    print(f"Checkpoint frequency: Every {args.save_frequency} questions")
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Checkpoints saved to: checkpoints/")
    
    # Show checkpoint status
    final_checkpoints = check_checkpoints()
    if final_checkpoints:
        print(f"⚠️  Remaining checkpoints: {final_checkpoints}")
        print("   (These indicate incomplete runs that can be resumed)")
    else:
        print("✅ All benchmarks completed successfully")
    
    # Exit with error code if any benchmark failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main() 