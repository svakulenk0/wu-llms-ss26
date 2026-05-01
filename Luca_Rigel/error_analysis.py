import os
import pandas as pd
from rouge_score import rouge_scorer

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "Austrian Tax Law Dataset - Dataset.csv")

if not os.path.exists(GROUND_TRUTH_PATH):
    print(f"Error: {GROUND_TRUTH_PATH} nicht gefunden.")
    exit(1)

gt_df = pd.read_csv(GROUND_TRUTH_PATH).dropna(subset=['id', 'correct_answer'])
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

def analyze_worst(file_path):
    if not os.path.exists(file_path):
        return
    df = pd.read_csv(file_path)
    results = []
    
    empty_count = 0
    refusals = ['weiß nicht', 'unbekannt', "don't know", 'keine informationen']
    refusal_count = 0
    
    for _, row in gt_df.iterrows():
        q_id = row['id']
        ref = str(row['correct_answer']).strip()
        pred_df = df[df['id'] == q_id]
        if pred_df.empty: continue
        pred = str(pred_df.iloc[0]['answer']).strip()
        
        if not pred or pred.lower() == 'nan':
            empty_count += 1
            score = 0.0
        else:
            if any(r in pred.lower() for r in refusals):
                refusal_count += 1
                
            score = scorer.score(ref, pred)['rougeL'].fmeasure
            
        results.append((score, q_id, ref, pred))
    
    results.sort(key=lambda x: x[0])
    
    print(f"\n=============================================")
    print(f"   ERROR ANALYSIS: {os.path.basename(file_path)}")
    print(f"=============================================")
    print(f"Total evaluated: {len(results)}")
    print(f"Empty answers (nan) count : {empty_count}")
    print(f"Refusal/Safe answers count: {refusal_count}")
    print(f"\n--- Top 3 Worst Predictions (by ROUGE-L) ---")
    for score, qid, ref, pred in results[:3]:
        print(f"ID: {qid} | Score: {score:.3f}")
        print(f"REF : {ref[:150]}...")
        print(f"PRED: {pred[:150]}...")
        print("-" * 50)

analyze_worst(os.path.join(BASE_DIR, "results", "model_1_output_FINAL.csv"))
analyze_worst(os.path.join(BASE_DIR, "results", "model_2_output_FINAL.csv"))
analyze_worst(os.path.join(BASE_DIR, "results", "model_3_output_FINAL.csv"))
