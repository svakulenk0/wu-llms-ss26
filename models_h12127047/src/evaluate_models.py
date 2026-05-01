import evaluate
import pandas as pd
import os
import glob

# Path configurations
ANSWERS_PATH = "answers.csv"
RESULTS_DIR = "results"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "evaluation_scores.csv")

def evaluate_results():
    # Load ground truth answers
    if not os.path.exists(ANSWERS_PATH):
        print(f"Error: {ANSWERS_PATH} not found.")
        return

    print(f"Loading ground truth from {ANSWERS_PATH}...")
    df_answers = pd.read_csv(ANSWERS_PATH)
    # Ensure mandatory columns exist
    if 'id' not in df_answers.columns or 'correct_answer' not in df_answers.columns:
        print("Error: answers.csv must contain 'id' and 'correct_answer' columns.")
        return
    
    df_answers = df_answers[['id', 'correct_answer']]

    # Find all result CSV files in results/
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory {RESULTS_DIR} not found.")
        return
        
    result_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
    # Filter out any summary files to avoid evaluating the output of this script
    result_files = [f for f in result_files if os.path.basename(f) != os.path.basename(OUTPUT_PATH)]

    if not result_files:
        print(f"No result files found in {RESULTS_DIR}.")
        return

    print(f"Found {len(result_files)} result files to evaluate.")

    # Initialize metrics
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        print(f"Error loading rouge metric: {e}")
        return

    summary_results = []

    for file_path in result_files:
        file_name = os.path.basename(file_path)
        print(f"\nProcessing {file_name}...")
        
        try:
            # Load model results
            df_results = pd.read_csv(file_path)
            
            # Check for required columns
            # Handle both 'model_answer' and 'answer'
            if 'model_answer' not in df_results.columns:
                if 'answer' in df_results.columns:
                    df_results = df_results.rename(columns={'answer': 'model_answer'})
                else:
                    print(f"  Skipping {file_name}: Missing 'id', 'model_answer', or 'answer' column.")
                    continue

            if 'id' not in df_results.columns:
                print(f"  Skipping {file_name}: Missing 'id' column.")
                continue

            # Merge with ground truth on ID
            df_merged = pd.merge(df_results, df_answers, on="id", how="inner")
            
            if df_merged.empty:
                print(f"  Warning: No matching IDs found in {file_name} compared to answers.csv.")
                continue

            # Perform evaluation
            metrics = rouge.compute(
                predictions=df_merged['model_answer'].fillna("").tolist(), 
                references=df_merged['correct_answer'].fillna("").tolist(), 
                use_stemmer=True
            )
            
            # Extract ROUGE-L score (evaluate library returns a flat dict)
            score = metrics['rougeL']
            matched_count = len(df_merged)
            print(f"  ROUGE-L Score: {score:.4f} (matched {matched_count} questions)")
            
            summary_results.append({
                "model_version": file_name.replace(".csv", ""),
                "rouge_l": round(float(score), 4),
                "matched_records": matched_count
            })
            
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")

    if summary_results:
        # Save results to evaluation_scores.csv
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSuccessfully saved evaluation summary to {OUTPUT_PATH}")

        # Print summary table
        print("\n" + "="*40)
        print("       EVALUATION SUMMARY")
        print("="*40)
        print(df_summary.to_string(index=False))
        print("="*40)
    else:
        print("\nNo evaluation results to save.")

if __name__ == "__main__":
    evaluate_results()