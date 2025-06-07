# AGIR Dental Benchmark

This directory contains the benchmark implementation for testing local AGIR models on the dental subset of the MedMCQA dataset.

## Overview

The AGIR benchmark is designed to test your local AGIR model (running at `http://localhost:8000`) on medical multiple-choice questions from the dental subject area.

## Requirements

1. **Local AGIR API Server**: Make sure your AGIR model is running at `http://localhost:8000`
2. **API Endpoint**: The benchmark expects a `/api/completions/simple` endpoint
3. **Dependencies**: Install required Python packages (same as other benchmarks)

## API Configuration

The benchmark is configured to use:
- **Base URL**: `http://localhost:8000`
- **Endpoint**: `/api/completions/simple`
- **User ID**: `e030d930-913d-4525-8478-1cf77b698364`

You can modify these settings in `agir_dental_benchmark.py` if your setup is different.

## Usage

### From the project root directory:

```bash
# Run all remaining questions
make agir

# Run 5 additional questions (continues from where you left off)
make agir LIMIT=5

# Run 1 question for testing
make agir LIMIT=1

# Clear all results and start fresh
make agir-clear

# Show current progress
cd src/dental/agir && python agir_dental_benchmark.py --progress

# Show performance metrics
cd src/dental/agir && python agir_dental_benchmark.py --metrics
```

### Direct usage:

```bash
cd src/dental/agir

# Run all remaining questions
python agir_dental_benchmark.py

# Run with specific options
python agir_dental_benchmark.py --limit 10 --delay 0.2

# Clear results and start fresh
python agir_dental_benchmark.py --clear-results

# Show progress
python agir_dental_benchmark.py --progress

# Show metrics
python agir_dental_benchmark.py --metrics
```

## Features

- **Checkpoint Support**: Automatically saves progress and can resume from interruptions
- **Progress Tracking**: Shows processing speed and estimated time remaining
- **Multiple Output Formats**: Saves results in both JSON and CSV formats
- **Error Recovery**: Handles API failures with retries
- **Performance Metrics**: Calculates accuracy and answer distribution

## Output Files

Results are saved in:
- `results/dental/agir/agir_results.json` - Complete results with metadata
- `results/dental/agir/agir_results.csv` - CSV format for analysis
- `results/dental/agir/agir_progress.json` - Progress tracking data

## API Request Format

The benchmark sends requests to your local API with this format:

```json
{
  "prompt": "You are a medical expert. Please answer the following multiple-choice question...",
  "max_tokens": 10,
  "temperature": 0.1,
  "user_id": "e030d930-913d-4525-8478-1cf77b698364"
}
```

## Expected API Response Format

Your API should return responses in this format:

```json
{
  "choices": [
    {
      "text": "A"
    }
  ]
}
```

The benchmark expects the model to respond with just the letter choice (A, B, C, or D).

## Troubleshooting

1. **Connection Failed**: Make sure your AGIR server is running at `http://localhost:8000`
2. **API Format Issues**: Check that your API endpoint matches the expected request/response format
3. **Slow Processing**: Adjust the `--delay` parameter to reduce API call frequency
4. **Resume Issues**: Delete the progress file if checkpoint recovery fails

## Performance Tips

- Use `--delay 0.1` (default) to avoid overwhelming your local API
- Monitor your server resources during benchmarking
- Use `--limit` for testing before running the full benchmark
- Check progress periodically with `--progress` flag 