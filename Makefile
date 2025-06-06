# Makefile for MedMCQA Dataset Processing

# Variables
PYTHON := python3
SCRIPT := src/dataset_process/seperate_dataset_by_subject.py
OUTPUT_DIR := datasets_by_subject

# Default target
.PHONY: help
help:
	@echo "MedMCQA Dataset Processing Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help          - Show this help message"
	@echo "  install       - Install required dependencies"
	@echo "  all           - Process all splits (train, test, validation)"
	@echo "  train         - Process only train split"
	@echo "  test          - Process only test split"
	@echo "  validation    - Process only validation split"
	@echo "  valid         - Alias for validation"
	@echo "  clean         - Remove output directory"
	@echo "  show-subjects - Show available subjects in dataset"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  benchmark-dental        - Run dental benchmarks for all models"
	@echo "  benchmark-dental-gpt41  - Run benchmark for GPT-4.1-nano only"
	@echo "  benchmark-dental-o3     - Run benchmark for O3-mini only"
	@echo "  benchmark-dental-test   - Run test benchmark (5 questions)"
	@echo "  benchmark-dental-csv    - Run benchmark with CSV export"
	@echo "  benchmark-clean         - Clean benchmark files"
	@echo "  benchmark-status        - Check benchmark progress"
	@echo ""
	@echo "Examples:"
	@echo "  make train              # Process only train split"
	@echo "  make test validation    # Process test and validation splits"
	@echo "  make OUTPUT_DIR=my_data all  # Use custom output directory"
	@echo "  make benchmark-dental   # Run dental AI benchmarks"

# Install dependencies
.PHONY: install
install:
	pip install datasets

# Process all splits
.PHONY: all
all:
	$(PYTHON) $(SCRIPT) --splits all --output-dir $(OUTPUT_DIR)

# Process individual splits
.PHONY: train
train:
	$(PYTHON) $(SCRIPT) --splits train --output-dir $(OUTPUT_DIR)

.PHONY: test
test:
	$(PYTHON) $(SCRIPT) --splits test --output-dir $(OUTPUT_DIR)

.PHONY: validation valid
validation valid:
	$(PYTHON) $(SCRIPT) --splits validation --output-dir $(OUTPUT_DIR)

# Process multiple splits (can be combined)
.PHONY: train-test
train-test:
	$(PYTHON) $(SCRIPT) --splits train test --output-dir $(OUTPUT_DIR)

.PHONY: train-valid
train-valid:
	$(PYTHON) $(SCRIPT) --splits train validation --output-dir $(OUTPUT_DIR)

.PHONY: test-valid
test-valid:
	$(PYTHON) $(SCRIPT) --splits test validation --output-dir $(OUTPUT_DIR)

# Show subjects without processing
.PHONY: show-subjects
show-subjects:
	@echo "Loading dataset to show available subjects..."
	@$(PYTHON) -c "from datasets import load_dataset; ds = load_dataset('openlifescienceai/medmcqa'); subjects = set(); [subjects.add(sample['subject_name']) for split in ['train', 'test', 'validation'] for sample in ds[split]]; print('\\nAvailable subjects:'); [print(f'  - {s}') for s in sorted(subjects)]"

# Clean output directory
.PHONY: clean
clean:
	@echo "Removing $(OUTPUT_DIR) directory..."
	rm -rf $(OUTPUT_DIR)
	@echo "Clean completed."

# Dental benchmark targets
.PHONY: benchmark-deps
benchmark-deps:
	@echo "Installing dental benchmark dependencies..."
	pip install -r src/dental/requirements.txt

.PHONY: benchmark-dental
benchmark-dental: benchmark-deps
	@echo "Running dental benchmark for all models..."
	cd src/dental && python run_benchmarks.py

.PHONY: benchmark-dental-gpt41
benchmark-dental-gpt41: benchmark-deps
	@echo "Running dental benchmark for GPT-4.1-nano..."
	cd src/dental && python run_benchmarks.py --models gpt4.1-nano

.PHONY: benchmark-dental-o3
benchmark-dental-o3: benchmark-deps
	@echo "Running dental benchmark for O3-mini..."
	cd src/dental && python run_benchmarks.py --models o3-mini

.PHONY: benchmark-dental-test
benchmark-dental-test: benchmark-deps
	@echo "Running dental benchmark test (limited questions)..."
	cd src/dental && python run_benchmarks.py --limit 1

.PHONY: benchmark-dental-csv
benchmark-dental-csv: benchmark-deps
	@echo "Running dental benchmark with CSV export..."
	cd src/dental && python run_benchmarks.py --export-csv

.PHONY: benchmark-clean
benchmark-clean:
	@echo "Cleaning benchmark checkpoints and results..."
	cd src/dental && rm -rf checkpoints/ results/
	@echo "Benchmark clean completed."

.PHONY: benchmark-status
benchmark-status:
	@echo "Checking benchmark status..."
	@if [ -d "src/dental/checkpoints" ]; then \
		echo "Checkpoints found:"; \
		ls -la src/dental/checkpoints/; \
	else \
		echo "No checkpoints found."; \
	fi
	@if [ -d "src/dental/results" ]; then \
		echo "Results found:"; \
		ls -la src/dental/results/; \
	else \
		echo "No results found."; \
	fi

# Check if output directory exists
.PHONY: status
status:
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "Output directory '$(OUTPUT_DIR)' exists:"; \
		ls -la $(OUTPUT_DIR) | head -20; \
		echo ""; \
		echo "Total files: $$(ls -1 $(OUTPUT_DIR) | wc -l)"; \
	else \
		echo "Output directory '$(OUTPUT_DIR)' does not exist."; \
	fi

# Development/testing target - dry run
.PHONY: dry-run
dry-run:
	@echo "This would run: $(PYTHON) $(SCRIPT) --splits all --output-dir $(OUTPUT_DIR)"
	@echo "Script exists: $$(test -f $(SCRIPT) && echo 'Yes' || echo 'No')"
	@echo "Python version: $$($(PYTHON) --version)" 