# AGIR Hospital Experiments

This repository contains experiments and benchmarks for medical AI models using the MedMCQA dataset.

## Dataset Source

This project uses the **MedMCQA** dataset from HuggingFace:
- **Dataset**: `openlifescienceai/medmcqa`
- **Paper**: "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering"
- **Authors**: Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu
- **Conference**: Conference on Health, Inference, and Learning (CHIL) 2022

### Dataset Description
MedMCQA is a large-scale Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions:
- **194k+ high-quality** AIIMS & NEET PG entrance exam MCQs
- **21 medical subjects** covering 2.4k healthcare topics
- **Subjects include**: Anesthesia, Anatomy, Biochemistry, Dental, ENT, Forensic Medicine, Obstetrics and Gynecology, Medicine, Microbiology, Ophthalmology, Orthopedics, Pathology, Pediatrics, Pharmacology, Physiology, Psychiatry, Radiology, Skin, Preventive & Social Medicine, Surgery

### Data Splits
- **Train**: 182,822 questions (mock & online test series)
- **Test**: 6,150 questions (AIIMS PG exam MCQs, 1991-present)
- **Validation**: 4,183 questions (NEET PG exam MCQs, 2001-present)

## Project Structure

```
├── src/
│   ├── dataset_process/         # Dataset processing utilities
│   │   ├── seperate_dataset_by_subject.py  # Split dataset by medical subject
│   │   ├── README.md
│   │   └── MedMCQA.md          # Dataset documentation
│   └── dental/                 # Dental subject benchmarks
│       ├── gpt-4-1-nano/       # GPT-4-1-nano experiments
│       └── o3-mini/            # O3-mini experiments
├── datasets_by_subject/        # Generated subject-specific datasets
├── Makefile                    # Build automation
└── README.md
```

## Setup

### Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

## Usage

### Dataset Processing

Use the Makefile to process the MedMCQA dataset by medical subject:

```bash
# Install dependencies
make install

# Process all splits (train, test, validation)
make all

# Process specific splits
make train          # Only training data
make test           # Only test data
make validation     # Only validation data

# View available subjects
make show-subjects

# Check processing status
make status

# Clean output
make clean
```

### Dental Benchmarks

Run AI model benchmarks on dental questions:

```bash
# Run all models
make benchmark-dental

# Run specific models
make benchmark-dental-gpt41
make benchmark-dental-o3

# Run test with limited questions
make benchmark-dental-test

# Export results to CSV
make benchmark-dental-csv

# Check benchmark status
make benchmark-status

# Clean benchmark results
make benchmark-clean
```

### Custom Processing

You can also run the processing script directly:

```bash
# Process specific splits
python src/dataset_process/seperate_dataset_by_subject.py --splits train test --output-dir my_output

# Process all splits to custom directory
python src/dataset_process/seperate_dataset_by_subject.py --splits all --output-dir custom_datasets
```

## Benchmarks

### Dental Subject Benchmarks

The dental subject contains specialized medical questions related to dentistry. Current benchmark implementations:

- **GPT-4.1-nano**: OpenAI's latest nano model for medical QA
- **O3-mini**: OpenAI's optimized mini model for medical reasoning

Each model directory contains:
- Evaluation scripts
- Results analysis
- Performance metrics

## Citation

If you use this work, please cite the original MedMCQA paper:

```bibtex
@InProceedings{pmlr-v174-pal22a,
  title = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  pages = {248--260},
  year = {2022},
  editor = {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = {174},
  series = {Proceedings of Machine Learning Research},
  month = {07--08 Apr},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v174/pal22a.html}
}
```

## License

This project follows the licensing terms of the original MedMCQA dataset. Please refer to the dataset's official documentation for license details.
