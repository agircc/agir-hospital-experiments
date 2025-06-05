from datasets import load_dataset
import os
from collections import defaultdict

ds = load_dataset("openlifescienceai/medmcqa")

# Create output directory if it doesn't exist
output_dir = "datasets_by_subject"
os.makedirs(output_dir, exist_ok=True)

# Get all unique subjects from the dataset
all_subjects = set()
for split in ['train', 'test', 'validation']:
    for sample in ds[split]:
        all_subjects.add(sample['subject_name'])

print(f"Found {len(all_subjects)} unique subjects:")
for subject in sorted(all_subjects):
    print(f"  - {subject}")

# Split dataset by subject for each split (train, test, validation)
for split_name in ['train', 'test', 'validation']:
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
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"  Saved {len(samples)} samples for {subject} to {filename}")

print(f"\nDataset splitting completed! Files saved in '{output_dir}' directory.")
print("\nSummary:")
for subject in sorted(all_subjects):
    clean_subject = subject.lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '')
    print(f"  {subject}:")
    print(f"    - {clean_subject}_train.jsonl")
    print(f"    - {clean_subject}_test.jsonl") 
    print(f"    - {clean_subject}_valid.jsonl")