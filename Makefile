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
	@echo "  gpt-4-1-nano           - Run GPT-4.1-nano benchmark"
	@echo "  gpt-o3                 - Run O3-mini benchmark" 
	@echo "  benchmark-clean        - Clean benchmark results"
	@echo "  benchmark-status       - Check benchmark progress"
	@echo "  gpt-4-1-nano-clear     - Clear GPT-4.1-nano results and restart"
	@echo "  gpt-o3-clear           - Clear O3-mini results and restart"
	@echo ""
	@echo "Parameters:"
	@echo "  LIMIT=N                - Process N additional questions (e.g. make gpt-4-1-nano LIMIT=10)"
	@echo "                          If not specified, processes all remaining questions"
	@echo ""
	@echo "Examples:"
	@echo "  make train              # Process only train split"
	@echo "  make test validation    # Process test and validation splits"
	@echo "  make OUTPUT_DIR=my_data all  # Use custom output directory"
	@echo "  make gpt-4-1-nano       # Run GPT-4.1-nano benchmark on all remaining questions"
	@echo "  make gpt-4-1-nano LIMIT=1  # Process 1 additional question (continues from where left off)"
	@echo "  make gpt-o3 LIMIT=10    # Run O3-mini with 10 additional questions"

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

# GPT-4.1-nano benchmark
.PHONY: gpt-4-1-nano
gpt-4-1-nano:
	@echo "Running GPT-4.1-nano dental benchmark..."
	@if [ -n "$(LIMIT)" ]; then \
		echo "Limiting to $(LIMIT) additional questions..."; \
		cd src/dental/gpt-4-1-nano && python gpt4_nano_benchmark.py --limit $(LIMIT); \
	else \
		echo "Running all remaining questions..."; \
		cd src/dental/gpt-4-1-nano && python gpt4_nano_benchmark.py; \
	fi

# O3-mini benchmark  
.PHONY: gpt-o3
gpt-o3:
	@echo "Running O3-mini dental benchmark..."
	@if [ -n "$(LIMIT)" ]; then \
		echo "Limiting to $(LIMIT) additional questions..."; \
		cd src/dental/o3-mini && python o3_mini_benchmark.py --limit $(LIMIT); \
	else \
		echo "Running all remaining questions..."; \
		cd src/dental/o3-mini && python o3_mini_benchmark.py; \
	fi

# Clear results and restart benchmarks
.PHONY: gpt-4-1-nano-clear
gpt-4-1-nano-clear:
	@echo "Clearing GPT-4.1-nano results and restarting..."
	cd src/dental/gpt-4-1-nano && python gpt4_nano_benchmark.py --clear-results
	@echo "Results cleared. You can now run 'make gpt-4-1-nano' to start fresh."

.PHONY: gpt-o3-clear  
gpt-o3-clear: benchmark-deps
	@echo "Clearing O3-mini results and restarting..."
	cd src/dental/o3-mini && python o3_mini_benchmark.py --clear-results
	@echo "Results cleared. You can now run 'make gpt-o3' to start fresh."

.PHONY: benchmark-clean
benchmark-clean:
	@echo "Cleaning benchmark results..."
	rm -rf results/dental/
	@echo "Benchmark clean completed."

.PHONY: benchmark-status
benchmark-status:
	@echo "Checking benchmark status..."
	@if [ -d "results/dental" ]; then \
		echo "Results found:"; \
		ls -la results/dental/; \
		echo ""; \
		echo "Progress summary:"; \
		for file in results/dental/*.csv; do \
			if [ -f "$$file" ]; then \
				lines=$$(wc -l < "$$file"); \
				completed=$$((lines - 1)); \
				echo "  $$(basename $$file): $$completed questions completed"; \
			fi; \
		done; \
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