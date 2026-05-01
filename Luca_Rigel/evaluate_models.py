

import os
import pandas as pd
from tqdm import tqdm

try:
    from rouge_score import rouge_scorer
    import sacrebleu
    import bert_score
except ImportError:
    print("Fehlende Bibliotheken! Bitte installieren mit:")
    print("pip install rouge-score sacrebleu bert-score pandas tqdm")
    exit(1)

# ==============================================================================
# SCHRITT 1: Pfade konfigurieren
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "Austrian Tax Law Dataset - Dataset.csv")

MODEL_FILES = {
    "Model 1 (Zero-Shot)": os.path.join(BASE_DIR, "results", "model_1_output_FINAL.csv"),
    "Model 2 (Fine-Tuning)": os.path.join(BASE_DIR, "results", "model_2_output_FINAL.csv"),
    "Model 3 (RAG)": os.path.join(BASE_DIR, "results", "model_3_output_FINAL.csv"),
}

if not os.path.exists(GROUND_TRUTH_PATH):
    print(f"Fehler: Ground Truth Datensatz nicht gefunden unter {GROUND_TRUTH_PATH}")
    exit(1)

gt_df = pd.read_csv(GROUND_TRUTH_PATH)
gt_df = gt_df.dropna(subset=['id', 'correct_answer'])

# ==============================================================================
# SCHRITT 3: Bewertungs-Funktion (ROUGE, BLEU & BERTScore Calculation)
# ==============================================================================
def calculate_metrics(csv_path, model_name):
    if not os.path.exists(csv_path):
        return None, None, None
    
    df = pd.read_csv(csv_path)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    bleu_scores = []
    rouge_l_scores = []
    
    cands_for_bert = []
    refs_for_bert = []
    
    for _, row in gt_df.iterrows():
        q_id = row['id']
        reference = str(row['correct_answer'])
        
        pred_row = df[df['id'] == q_id]
        if pred_row.empty:
            continue
        
        prediction = str(pred_row.iloc[0]['answer'])
        if pd.isna(prediction) or prediction.strip() == "":
            continue
            
        rouge_score = scorer.score(reference, prediction)
        rouge_l_scores.append(rouge_score['rougeL'].fmeasure)
        
        bleu = sacrebleu.corpus_bleu([prediction], [[reference]])
        bleu_scores.append(bleu.score)
        
        cands_for_bert.append(prediction)
        refs_for_bert.append(reference)
        
    if not bleu_scores:
        return 0.0, 0.0, 0.0
        
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_l_scores) / len(rouge_l_scores)
    
    print(f"Berechne BERTScore für {model_name}...")
    P, R, F1 = bert_score.score(cands_for_bert, refs_for_bert, lang="de", verbose=False)
    avg_bert = F1.mean().item()
    
    return avg_bleu, avg_rouge, avg_bert

# ==============================================================================
# SCHRITT 4: Auswertung
# ==============================================================================
print("="*75)
print(" AUTOMATISIERTE EVALUATION (Austrian Tax Law)")
print(f" Datensatz: {len(gt_df)} Ground-Truth Antworten")
print("="*75)
print(f"{'Modell':<25} | {'BLEU Score':<12} | {'ROUGE-L (F1)':<12} | {'BERTScore':<10}")
print("-" * 75)

for model_name, path in MODEL_FILES.items():
    bleu, rouge, bert = calculate_metrics(path, model_name)
    if bleu is None:
        print(f"{model_name:<25} | Datei nicht gefunden!")
    else:
        print(f"{model_name:<25} | {bleu:12.2f} | {rouge:12.4f} | {bert:10.4f}")

print("-" * 75)
print("Hinweis: BLEU (0-100), ROUGE-L (0-1.0), BERTScore (0-1.0).")
print("BERTScore erfasst semantische Ähnlichkeit (besser für Generative LLMs).")
print("="*75)
