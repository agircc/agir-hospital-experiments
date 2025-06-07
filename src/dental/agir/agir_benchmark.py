import os
import csv
import time
import json
import argparse
import datetime
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Configuration
LOCAL_API_BASE = "http://localhost:8000"
USER_ID = "e030d930-913d-4525-8478-1cf77b698364"

# File paths
DATASET_PATH = 'dataset/dreaddit-test.csv'
RESULTS_DIR = 'results'
MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'agir-after-cases')
RESULTS_FILE = os.path.join(MODEL_RESULTS_DIR, 'result.csv')
PROGRESS_FILE = os.path.join(MODEL_RESULTS_DIR, 'progress.json')

def ensure_dirs():
    """Ensure the required directories exist."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def get_last_processed_id():
    """Get the ID of the last processed entry from the results CSV."""
    if not os.path.exists(RESULTS_FILE):
        return None
    
    try:
        with open(RESULTS_FILE, 'r', newline='', encoding='latin-1') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            last_id = None
            for row in reader:
                if row:  # Make sure row is not empty
                    last_id = row[0]  # First column is ID
            return last_id
    except Exception as e:
        print(f"Error reading results file: {e}")
        return None

def create_prompt(text: str) -> str:
    """Create a prompt for the model."""
    return f"""You are an expert tasked with analyzing a piece of text to assess signs of emotional distress.

Instructions:
- Carefully read the following text.
- Determine whether it suggests the presence of stress, anxiety, trauma, or emotional distress.
- Respond **strictly** with one of the following:
  - "1" if the text indicates stress, anxiety, trauma, or distress.
  - "0" if the text appears neutral and does not indicate significant emotional distress.

Important:
- Only reply with a single character: "0" or "1".
- Do not include any explanation, commentary, or additional text.

Text to analyze:
{text}

Your classification ("0" or "1"):"""

def call_local_api(prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call the local API to get a completion."""
    url = f"{LOCAL_API_BASE}/api/completions/simple"
    
    payload = {
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0.1,
        "user_id": USER_ID
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the text response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['text'].strip()
            else:
                print(f"Unexpected response format: {result}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                print(f"Failed to get response after {max_retries} attempts")
                return None
        except Exception as e:
            print(f"Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None
    
    return None

def process_entry(entry: Dict[str, Any]) -> Optional[Tuple[str, str, str, bool]]:
    """Process a single entry and return the result."""
    entry_id = entry['id']
    text = entry['text']
    actual_label = entry['label']
    
    # Create prompt and call API
    prompt = create_prompt(text)
    response = call_local_api(prompt)
    
    if response is None:
        print(f"Failed to get response for entry {entry_id}")
        return None
    
    # Extract the predicted label
    predicted_label = None
    if '0' in response:
        predicted_label = '0'
    elif '1' in response:
        predicted_label = '1'
    else:
        # If model didn't give a clear 0 or 1, try to infer from text
        if any(word in response.lower() for word in ['stress', 'distress', 'anxiety', 'trauma', 'yes']):
            predicted_label = '1'
        else:
            predicted_label = '0'
        print(f"Warning: Unclear response for ID {entry_id}: '{response}', inferred as {predicted_label}")
    
    is_correct = predicted_label == actual_label
    
    return (entry_id, actual_label, predicted_label, is_correct)

def save_result(result: Tuple[str, str, str, bool]):
    """Save a single result to the CSV file."""
    file_exists = os.path.exists(RESULTS_FILE)
    
    try:
        with open(RESULTS_FILE, 'a', newline='', encoding='latin-1') as f:
            writer = csv.writer(f)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['id', 'actual_label', 'predicted_label', 'is_correct'])
            
            # Write result
            entry_id, actual_label, predicted_label, is_correct = result
            writer.writerow([
                entry_id, 
                actual_label, 
                predicted_label, 
                '1' if is_correct else '0'
            ])
        
        print(f"Saved result for entry {result[0]}")
    except Exception as e:
        print(f"Error saving result: {e}")

def save_progress(processed_count: int, total_count: int, start_time: float):
    """Save progress information to JSON file."""
    progress_data = {
        "processed_count": processed_count,
        "total_count": total_count,
        "start_time": start_time,
        "last_update": time.time(),
        "progress_percentage": round((processed_count / total_count) * 100, 2) if total_count > 0 else 0
    }
    
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress() -> Optional[Dict]:
    """Load progress information from JSON file."""
    if not os.path.exists(PROGRESS_FILE):
        return None
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading progress: {e}")
        return None

def test_api_connection():
    """Test if the local API is accessible."""
    print("Testing API connection...")
    try:
        test_prompt = "This is a test prompt."
        response = call_local_api(test_prompt, max_retries=1)
        if response is not None:
            print("✓ API connection successful")
            return True
        else:
            print("✗ API connection failed")
            return False
    except Exception as e:
        print(f"✗ API connection error: {e}")
        return False

def calculate_metrics():
    """Calculate and display performance metrics from results file."""
    if not os.path.exists(RESULTS_FILE):
        print("No results file found")
        return
    
    try:
        correct_count = 0
        total_count = 0
        
        with open(RESULTS_FILE, 'r', newline='', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_count += 1
                if row['is_correct'] == '1':
                    correct_count += 1
        
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print(f"\nPerformance Metrics:")
            print(f"Total processed: {total_count}")
            print(f"Correct predictions: {correct_count}")
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            print("No results found in file")
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")

def main():
    """Main function to process the dataset using the local API."""
    parser = argparse.ArgumentParser(description='Process Dreaddit dataset using local agir-learner API')
    parser.add_argument('--test-entries', type=int, default=0, help='Number of entries to test (0 for all entries)')
    parser.add_argument('--encoding', type=str, default='latin-1', help='Encoding to use for reading CSV files')
    parser.add_argument('--metrics', action='store_true', help='Calculate and display metrics from existing results')
    parser.add_argument('--progress', action='store_true', help='Show current progress')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between API calls in seconds')
    args = parser.parse_args()
    
    ensure_dirs()
    
    # If metrics are requested
    if args.metrics:
        calculate_metrics()
        return
    
    # If progress is requested
    if args.progress:
        progress = load_progress()
        if progress:
            print(f"Progress: {progress['processed_count']}/{progress['total_count']} ({progress['progress_percentage']}%)")
            elapsed = time.time() - progress['start_time']
            print(f"Elapsed time: {elapsed/3600:.2f} hours")
            if progress['processed_count'] > 0:
                avg_time = elapsed / progress['processed_count']
                remaining = (progress['total_count'] - progress['processed_count']) * avg_time
                print(f"Estimated remaining time: {remaining/3600:.2f} hours")
        else:
            print("No progress data found")
        return
    
    # Test API connection first
    if not test_api_connection():
        print("Cannot connect to local API. Please ensure the server is running at http://localhost:8000")
        return
    
    try:
        # Get last processed ID and continue
        last_processed_id = get_last_processed_id()
        print(f"Resuming from ID: {last_processed_id}" if last_processed_id else "Starting fresh")
        
        # Flag to indicate if we found the last processed ID in the dataset
        found_last_id = last_processed_id is None
        
        # Read the dataset
        entries = []
        
        # Try to open the file with the specified encoding
        try:
            with open(DATASET_PATH, 'r', newline='', encoding=args.encoding) as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    entry_id = row['id']
                    
                    # Skip entries until we find the last processed ID
                    if not found_last_id:
                        if entry_id == last_processed_id:
                            found_last_id = True
                        continue
                    
                    entry = {
                        "id": entry_id,
                        "text": row['text'],
                        "label": row['label']
                    }
                    
                    entries.append(entry)
                    
                    # Limit entries if testing
                    if args.test_entries > 0 and len(entries) >= args.test_entries:
                        break
                        
        except UnicodeDecodeError as e:
            print(f"Error reading CSV with {args.encoding} encoding: {e}")
            print("Try specifying a different encoding with --encoding parameter.")
            print("Common encodings to try: latin-1, iso-8859-1, cp1252")
            return
        
        if not entries:
            print("No entries to process")
            return
        
        print(f"Processing {len(entries)} entries")
        if args.test_entries > 0:
            print(f"(Testing mode: limited to {args.test_entries} entries)")
        
        # Initialize progress tracking
        start_time = time.time()
        processed_count = 0
        total_count = len(entries)
        
        # Process entries sequentially
        for i, entry in enumerate(entries):
            print(f"Processing entry {i+1}/{len(entries)} (ID: {entry['id']})")
            
            result = process_entry(entry)
            
            if result:
                save_result(result)
                processed_count += 1
                
                # Show progress every 10 entries
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (len(entries) - i - 1) * avg_time
                    print(f"Progress: {i+1}/{len(entries)} ({((i+1)/len(entries)*100):.1f}%) - "
                          f"ETA: {remaining/60:.1f} minutes")
                    
                    # Save progress
                    save_progress(processed_count, total_count, start_time)
            else:
                print(f"Failed to process entry {entry['id']}")
            
            # Add delay between requests to avoid overwhelming the API
            if args.delay > 0:
                time.sleep(args.delay)
        
        print(f"\nCompleted processing {processed_count}/{total_count} entries")
        
        # Calculate final metrics
        calculate_metrics()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
        # Save current progress
        if 'processed_count' in locals() and 'total_count' in locals() and 'start_time' in locals():
            save_progress(processed_count, total_count, start_time)

if __name__ == "__main__":
    main()
