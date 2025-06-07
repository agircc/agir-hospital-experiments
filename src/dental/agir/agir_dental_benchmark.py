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
USER_ID = "544b17ee-0aa3-44c6-b14c-7a67d21f5ecd"

# File paths - using absolute path detection like other benchmarks
def find_project_root():
    """Find the project root directory by looking for .git or Makefile."""
    current_dir = Path(__file__).resolve().parent
    while current_dir != current_dir.parent:
        if (current_dir / '.git').exists() or (current_dir / 'Makefile').exists():
            return current_dir
        current_dir = current_dir.parent
    return Path(__file__).resolve().parent

PROJECT_ROOT = find_project_root()
DATASET_PATH = PROJECT_ROOT / 'datasets_by_subject' / 'dental_valid.jsonl'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'dental' / 'agir_v4'
RESULTS_FILE = RESULTS_DIR / 'agir_results.json'
CSV_FILE = RESULTS_DIR / 'agir_results.csv'
PROGRESS_FILE = RESULTS_DIR / 'agir_progress.json'

def ensure_dirs():
    """Ensure the required directories exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset() -> List[Dict[str, Any]]:
    """Load the dental test dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    questions = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                questions.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
    
    print(f"Loaded {len(questions)} questions from dental dataset")
    return questions

def get_last_processed_index() -> int:
    """Get the index of the last processed question."""
    if not PROGRESS_FILE.exists():
        return 0
    
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            return progress.get('last_processed_index', 0)
    except Exception as e:
        print(f"Error reading progress file: {e}")
        return 0

def create_prompt(question_data: Dict[str, Any]) -> str:
    """Create a prompt for the medical question."""
    question = question_data['question']
    option_a = question_data['opa']
    option_b = question_data['opb'] 
    option_c = question_data['opc']
    option_d = question_data['opd']
    
    return f"""You are a medical expert. Please answer the following multiple-choice question by selecting the best answer.

Question: {question}

A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Instructions:
- Carefully read the question and all options
- Select the most accurate answer based on medical knowledge
- Respond with ONLY the letter of your choice: A, B, C, or D
- Do not include any explanation or additional text

Your answer:"""

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

def extract_answer(response: str) -> str:
    """Extract the answer choice from the model response."""
    if not response:
        return "INVALID"
    
    response = response.strip().upper()
    
    # Look for clear A, B, C, D answers
    for option in ['A', 'B', 'C', 'D']:
        if option in response:
            return option
    
    print(f"Warning: Could not extract clear answer from response: '{response}'")
    return "INVALID"

def process_question(question_data: Dict[str, Any], question_index: int) -> Optional[Dict[str, Any]]:
    """Process a single question and return the result."""
    question_id = question_data.get('id', f"q_{question_index}")
    
    # Create prompt and call API
    prompt = create_prompt(question_data)
    response = call_local_api(prompt)
    
    if response is None:
        print(f"Failed to get response for question {question_id}")
        return None
    
    # Extract the predicted answer
    predicted_answer = extract_answer(response)
    
    # Get correct answer - convert cop (0-3) to letter (A-D)
    correct_option_num = question_data.get('cop', -1)
    if correct_option_num in [0, 1, 2, 3]:
        correct_answer = chr(ord('A') + correct_option_num)  # 0->A, 1->B, 2->C, 3->D
    else:
        print(f"Warning: Invalid correct option {correct_option_num} for question {question_id}")
        correct_answer = "UNKNOWN"
    
    is_correct = predicted_answer == correct_answer and predicted_answer != "INVALID"
    
    result = {
        'question_id': question_id,
        'question_index': question_index,
        'question': question_data['question'],
        'options': {
            'A': question_data['opa'],
            'B': question_data['opb'], 
            'C': question_data['opc'],
            'D': question_data['opd']
        },
        'correct_answer': correct_answer,
        'predicted_answer': predicted_answer,
        'is_correct': is_correct,
        'raw_response': response,
        'timestamp': datetime.datetime.now().isoformat(),
        'subject': question_data.get('subject_name', 'dental')
    }
    
    return result

def save_result(result: Dict[str, Any]):
    """Save a single result to both JSON and CSV files."""
    # Save to JSON
    results = []
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                results = json.load(f)
        except:
            results = []
    
    results.append(result)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save to CSV
    file_exists = CSV_FILE.exists()
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'question_id', 'question_index', 'question', 'option_a', 'option_b', 
                'option_c', 'option_d', 'correct_answer', 'predicted_answer', 
                'is_correct', 'raw_response', 'timestamp', 'subject'
            ])
        
        # Write result
        writer.writerow([
            result['question_id'],
            result['question_index'],
            result['question'],
            result['options']['A'],
            result['options']['B'],
            result['options']['C'],
            result['options']['D'],
            result['correct_answer'],
            result['predicted_answer'],
            '1' if result['is_correct'] else '0',
            result['raw_response'],
            result['timestamp'],
            result['subject']
        ])

def save_progress(processed_count: int, total_count: int, start_time: float, last_processed_index: int):
    """Save progress information to JSON file."""
    progress_data = {
        "processed_count": processed_count,
        "total_count": total_count,
        "start_time": start_time,
        "last_update": time.time(),
        "last_processed_index": last_processed_index,
        "progress_percentage": round((processed_count / total_count) * 100, 2) if total_count > 0 else 0
    }
    
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress() -> Optional[Dict]:
    """Load progress information from JSON file."""
    if not PROGRESS_FILE.exists():
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
    if not RESULTS_FILE.exists():
        print("No results file found")
        return
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        
        if not results:
            print("No results found")
            return
        
        total_count = len(results)
        correct_count = sum(1 for r in results if r['is_correct'])
        
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\nPerformance Metrics:")
        print(f"Total processed: {total_count}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Show breakdown by predicted answer
        answer_counts = {}
        for result in results:
            pred = result['predicted_answer']
            if pred not in answer_counts:
                answer_counts[pred] = {'total': 0, 'correct': 0}
            answer_counts[pred]['total'] += 1
            if result['is_correct']:
                answer_counts[pred]['correct'] += 1
        
        print(f"\nAnswer Distribution:")
        for answer in sorted(answer_counts.keys()):
            counts = answer_counts[answer]
            acc = (counts['correct'] / counts['total'] * 100) if counts['total'] > 0 else 0
            print(f"  {answer}: {counts['total']} questions ({acc:.1f}% accuracy)")
            
    except Exception as e:
        print(f"Error calculating metrics: {e}")

def main():
    """Main function to process the dental dataset using the local API."""
    parser = argparse.ArgumentParser(description='Process dental dataset using local agir-learner API')
    parser.add_argument('--limit', type=int, default=0, help='Number of questions to process (0 for all remaining)')
    parser.add_argument('--metrics', action='store_true', help='Calculate and display metrics from existing results')
    parser.add_argument('--progress', action='store_true', help='Show current progress')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between API calls in seconds')
    parser.add_argument('--clear-results', action='store_true', help='Clear existing results and start fresh')
    args = parser.parse_args()
    
    ensure_dirs()
    
    # Clear results if requested
    if args.clear_results:
        for file_path in [RESULTS_FILE, CSV_FILE, PROGRESS_FILE]:
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Cleared {file_path}")
        print("All results cleared. You can now run the benchmark fresh.")
        return
    
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
        # Load dataset
        questions = load_dataset()
        
        if not questions:
            print("No questions found in dataset")
            return
        
        # Get last processed index and continue
        last_processed_index = get_last_processed_index()
        start_index = last_processed_index
        
        print(f"Dataset has {len(questions)} questions")
        if start_index > 0:
            print(f"Resuming from question {start_index + 1}")
        
        # Determine which questions to process
        questions_to_process = questions[start_index:]
        
        # Apply limit if specified
        if args.limit > 0:
            questions_to_process = questions_to_process[:args.limit]
            print(f"Limited to {args.limit} questions")
        
        if not questions_to_process:
            print("No questions to process")
            return
        
        print(f"Processing {len(questions_to_process)} questions")
        
        # Initialize progress tracking
        start_time = time.time()
        processed_count = 0
        total_count = len(questions_to_process)
        
        # Process questions sequentially
        for i, question_data in enumerate(questions_to_process):
            current_index = start_index + i
            print(f"Processing question {i+1}/{len(questions_to_process)} (Index: {current_index + 1})")
            
            result = process_question(question_data, current_index)
            
            if result:
                save_result(result)
                processed_count += 1
                
                # Show progress every 5 questions
                if (i + 1) % 5 == 0 or i == len(questions_to_process) - 1:
                    elapsed = time.time() - start_time
                    if processed_count > 0:
                        avg_time = elapsed / processed_count
                        remaining_questions = len(questions_to_process) - (i + 1)
                        remaining_time = remaining_questions * avg_time
                        print(f"Progress: {i+1}/{len(questions_to_process)} ({((i+1)/len(questions_to_process)*100):.1f}%) - "
                              f"ETA: {remaining_time/60:.1f} minutes")
                    
                    # Save progress
                    save_progress(processed_count, len(questions), start_time, current_index)
            else:
                print(f"Failed to process question at index {current_index}")
            
            # Add delay between requests to avoid overwhelming the API
            if args.delay > 0:
                time.sleep(args.delay)
        
        print(f"\nCompleted processing {processed_count}/{total_count} questions")
        
        # Calculate final metrics
        calculate_metrics()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
        # Save current progress
        if 'processed_count' in locals() and 'start_time' in locals() and 'current_index' in locals():
            save_progress(processed_count, len(questions), start_time, current_index)

if __name__ == "__main__":
    main() 