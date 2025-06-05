from datasets import load_dataset
import os
import sys
import argparse
from collections import defaultdict
import json

def main():
    parser = argparse.ArgumentParser(description='Split MedMCQA dataset by subject')
    parser.add_argument('--splits', nargs='+', choices=['train', 'test', 'validation', 'all'],
                      default=['all'], help='Which splits to process (default: all)')
    parser.add_argument('--output-dir', default='datasets_by_subject',
                      help='Output directory for split datasets (default: datasets_by_subject)')
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.splits:
        splits_to_process = ['train', 'test', 'validation']
    else:
        splits_to_process = args.splits
    
    print(f"Loading MedMCQA dataset...")
    ds = load_dataset("openlifescienceai/medmcqa")
    
    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique subjects from the dataset
    all_subjects = set()
    for split in splits_to_process:
        if split in ds:
            for sample in ds[split]:
                all_subjects.add(sample['subject_name'])
    
    print(f"Found {len(all_subjects)} unique subjects:")
    for subject in sorted(all_subjects):
        print(f"  - {subject}")
    
    # Split dataset by subject for each specified split
    for split_name in splits_to_process:
        if split_name not in ds:
            print(f"Warning: Split '{split_name}' not found in dataset")
            continue
            
        print(f"\nProcessing {split_name} split...")
        
        # Group samples by subject
        subjects_data = defaultdict(list)
        for sample in ds[split_name]:
            subject = sample['subject_name']
            subjects_data[subject].append(sample)
        
        # Save each subject's data
        for subject, samples in subjects_data.items():
            # Clean subject name for filename (remove special characters)
            clean_subject = subject.lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
            
            # Create filename based on split
            if split_name == 'validation':
                filename = f"{clean_subject}_valid.jsonl"
            else:
                filename = f"{clean_subject}_{split_name}.jsonl"
            
            filepath = os.path.join(output_dir, filename)
            
            # Save samples to JSONL format
            with open(filepath, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"  Saved {len(samples)} samples for {subject} to {filename}")
    
    print(f"\nDataset splitting completed! Files saved in '{output_dir}' directory.")
    print(f"Processed splits: {', '.join(splits_to_process)}")
    
    if len(splits_to_process) == 3:  # Only show full summary if all splits were processed
        print("\nSummary:")
        for subject in sorted(all_subjects):
            clean_subject = subject.lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
            print(f"  {subject}:")
            print(f"    - {clean_subject}_train.jsonl")
            print(f"    - {clean_subject}_test.jsonl") 
            print(f"    - {clean_subject}_valid.jsonl")

if __name__ == "__main__":
    main()