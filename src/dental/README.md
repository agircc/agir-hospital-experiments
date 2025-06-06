# Dental Subject Benchmarks

This module provides benchmarking tools for evaluating AI models on dental-specific medical questions from the MedMCQA dataset.

## Overview

The dental benchmark evaluates two OpenAI models:
- **GPT-4-1-nano**: OpenAI's latest nano model for medical QA
- **O3-mini**: OpenAI's optimized mini model for medical reasoning

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

First, generate the dental test dataset:

```bash
# From project root directory
make test  # This will create datasets_by_subject/dental_test.jsonl
```

### 3. Set API Key

Both models use the same OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage

### Quick Start

Run all benchmarks:
```bash
python run_benchmarks.py
```

### Individual Model Testing

#### GPT-4-1-nano:
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
python run_benchmarks.py --models gpt4-nano
python run_benchmarks.py --models o3-mini
python run_benchmarks.py --models gpt4-nano o3-mini
```

#### Use Mock Responses for O3-mini:
```bash
python run_benchmarks.py --mock-o3
```

#### Custom Output Directory:
```bash
python run_benchmarks.py --output-dir my_results
```

#### Pass API Key Directly:
```bash
python run_benchmarks.py --openai-key sk-xxx
```

## Output

Results are saved as JSON files containing:
- **Model performance metrics** (accuracy, duration)
- **Individual question results** with predictions and explanations
- **Error analysis** for failed cases
- **Metadata** (timestamp, model configuration)

### Example Output Structure:
```json
{
  "model_name": "gpt-4-1-nano",
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

### Base Class: `benchmark_base.py`
- **DentalBenchmark**: Abstract base class with common functionality
- Question loading and formatting
- Answer extraction and evaluation
- Results compilation and saving

### Model Implementations:
- **`gpt-4-1-nano/gpt4_nano_benchmark.py`**: GPT-4-1-nano specific implementation
- **`o3-mini/o3_mini_benchmark.py`**: O3-mini specific implementation with mock support

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
from benchmark_base import DentalBenchmark

class NewModelBenchmark(DentalBenchmark):
    def __init__(self, api_key: str = None):
        super().__init__("new-model")
        self.api_key = api_key
    
    def query_model(self, prompt: str) -> str:
        # Implement your model's API call here
        return response
```

## Troubleshooting

### Common Issues:

1. **"Test data not found"**
   - Run `make test` from project root to generate dental test data

2. **API Key errors**
   - Set OPENAI_API_KEY environment variable or pass --openai-key argument
   - For O3-mini: use `--mock-o3` flag if API is not available

3. **Import errors**
   - Ensure you're running from the `src/dental/` directory
   - Install dependencies: `pip install -r requirements.txt`

### Debug Mode:
```bash
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python run_benchmarks.py --limit 1
```

## Performance Notes

- **GPT-4-1-nano**: ~1.5-2 seconds per question
- **O3-mini**: Varies based on API response time
- **Memory usage**: Minimal, results stored incrementally
- **Parallel processing**: Not implemented (to respect API rate limits)

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include logging for debugging
4. Update documentation for new features
5. Test with `--limit` flag before full runs 