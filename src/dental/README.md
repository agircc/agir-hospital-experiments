# Dental Subject Benchmarks

This module provides benchmarking tools for evaluating AI models on dental-specific medical questions from the MedMCQA dataset.

## Overview

The dental benchmark evaluates two OpenAI models:
- **GPT-4.1-nano**: OpenAI's latest nano model for medical QA
- **O3-mini**: OpenAI's optimized mini model for medical reasoning

## Results
GPT-4.1-nano Dental Benchmark Results:
- Model: gpt-4.1-nano (gpt-4.1-nano)
- Dataset Total: 1318 questions
- Total Completed: 1318/1318
- This Run: 1311 questions processed
- This Run Accuracy: 690/1311 (52.63%)
- Duration: 675.88 seconds


O3-mini Dental Benchmark Results:
- Model: o3-mini (o3-mini)
- Dataset Total: 1318 questions
- Total Completed: 1318/1318
- This Run: 1311 questions processed
- This Run Accuracy: 839/1311 (64.00%)
- Duration: 4790.66 seconds

agir:
- Total processed: 1318
- Correct predictions: 680
- Accuracy: 51.59%

agir v2:
- Total processed: 1318
- Correct predictions: 683
- Accuracy: 51.82%

agir_v3
- Total processed: 1318
- Correct predictions: 680
- Accuracy: 51.59%

agir_v4
- Total processed: 1318
- Correct predictions: 673
- Accuracy: 51.06%

agir_v5
- Total processed: 1318
- Correct predictions: 680
- Accuracy: 51.59%

agir_v6:
- Total processed: 1318
- Correct predictions: 700
- Accuracy: 53.11%

## Key Features

✅ **Checkpoint Support**: Automatically saves progress and can resume from interruptions  
✅ **Unified Codebase**: Both models share the same implementation to avoid code differences  
✅ **Progress Tracking**: Real-time progress monitoring and accuracy tracking  
✅ **Error Recovery**: Continues processing even if individual questions fail

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

First, generate the dental test dataset:

```bash
# From project root directory
make valid  # This will create datasets_by_subject/dental_valid.jsonl
```

### 3. Set API Key

Both models use the same OpenAI API key. You can either:

**Option 1: Environment variable**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Option 2: .env file (Recommended)**
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### Quick Start

Run all benchmarks:
```bash
python run_benchmarks.py
```

### Individual Model Testing

#### GPT-4.1-nano:
```bash
python gpt-4-1-nano/gpt4_nano_benchmark.py
```

#### O3-mini:
```bash
python o3-mini/o3_mini_benchmark.py
```

### Advanced Options

#### Test with Limited Questions (for development):
```bash
python run_benchmarks.py --limit 10
```

#### Run Specific Models:
```bash
python run_benchmarks.py --models gpt4.1-nano
python run_benchmarks.py --models o3-mini
python run_benchmarks.py --models gpt4.1-nano o3-mini
```

#### Checkpoint Management:
```bash
# Clear existing checkpoints and start fresh
python run_benchmarks.py --clear-checkpoints

# Continue from existing checkpoint (default)
python run_benchmarks.py

# Save checkpoint every 10 questions instead of default 5
python run_benchmarks.py --save-frequency 10
```

#### Custom Output Directory:
```bash
python run_benchmarks.py --output-dir my_results
```

#### Pass API Key Directly:
```bash
python run_benchmarks.py --openai-key sk-xxx
```

#### Export to CSV:
```bash
python run_benchmarks.py --export-csv
```

## Output

### Output Formats

**JSON (Default)** - Complete results with full model responses:
- Model performance metrics (accuracy, duration)
- Individual question results with predictions and explanations
- Error analysis for failed cases
- Metadata (timestamp, model configuration)
- Checkpoint support for resuming interrupted runs

**CSV (Optional)** - Simplified tabular format for analysis:
- Flattened data suitable for Excel/pandas
- Question summaries and results
- Performance metrics per question

### Files Generated
- **`results/model_results_timestamp.json`**: Complete JSON results
- **`results/model_results_timestamp.csv`**: CSV export (if requested)
- **`checkpoints/model_checkpoint.json`**: Full checkpoint with all progress
- **`checkpoints/model_progress.json`**: Summary of current progress

### Example Output Structure:
```json
{
  "model_name": "gpt-4.1-nano",
  "model_id": "gpt-4.1-nano",
  "total_questions": 150,
  "correct_answers": 120,
  "accuracy": 0.80,
  "duration_seconds": 245.6,
  "timestamp": "2024-01-15T10:30:00",
  "results": [
    {
      "question_id": "abc-123",
      "question": "A patient presents with...",
      "correct_option": 2,
      "predicted_answer": "B",
      "is_correct": true,
      "response": "Looking at this case...",
      "topic": "Oral Pathology",
      "subject": "Dental"
    }
  ]
}
```

## Architecture

### Base Classes:
- **`benchmark_base.py`**: Abstract base class with common functionality
- **`openai_benchmark_base.py`**: OpenAI-specific base with checkpoint support

### Model Implementations:
- **`gpt-4-1-nano/gpt4_nano_benchmark.py`**: GPT-4.1-nano implementation (inherits from OpenAI base)
- **`o3-mini/o3_mini_benchmark.py`**: O3-mini implementation (inherits from OpenAI base)

### Utilities:
- **`run_benchmarks.py`**: Orchestrates multiple model benchmarks
- **`requirements.txt`**: Python dependencies

## Development

### Adding New Models

1. Create a new directory: `src/dental/new-model/`
2. Implement a class inheriting from `DentalBenchmark`
3. Override the `query_model()` method
4. Update `run_benchmarks.py` to include the new model

### Example Implementation:
```python
from openai_benchmark_base import OpenAIBenchmark

class NewModelBenchmark(OpenAIBenchmark):
    def __init__(self, api_key: str = None):
        super().__init__("new-model", "new-model-id", api_key)
    
    # All functionality inherited - no additional code needed!
```

## Troubleshooting

### Common Issues:

1. **"Test data not found"**
   - Run `make test` from project root to generate dental test data

2. **API Key errors**
   - Set OPENAI_API_KEY environment variable or pass --openai-key argument

3. **Checkpoint issues**
   - Use `--clear-checkpoints` to start fresh if data has changed
   - Check `checkpoints/` directory for existing progress files

4. **Import errors**
   - Ensure you're running from the `src/dental/` directory
   - Install dependencies: `pip install -r requirements.txt`

### Debug Mode:
```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python run_benchmarks.py --limit 1
```

## Performance Notes

- **GPT-4.1-nano**: ~1.5-2 seconds per question
- **O3-mini**: Varies based on API response time
- **Memory usage**: Minimal, results stored incrementally
- **Checkpoint overhead**: ~50ms per save operation
- **Recovery time**: Instant resume from last checkpoint
- **Parallel processing**: Not implemented (to respect API rate limits)

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include logging for debugging
4. Update documentation for new features
5. Test with `--limit` flag before full runs 