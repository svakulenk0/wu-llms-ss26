"""
evaluation_gold.py - Stage 3: final stricter gold-label evaluation
(WU LLMs SS26, Team 11)

Part of the Team 11 Stage 3 evaluation pipeline:
  Stage 1 - evaluation.py        (broad proxy evaluation, all 643 Qs)
  Stage 2 - citation_check.py    (systematic citation validity, all 643 Qs)
  Stage 3 - evaluation_gold.py   (final gold-label evaluation, 60 Qs)
  Stage 4 - visualize_results.py (figures from the final CSVs)
Orchestrator: run_all_evaluations.py

Feeds the Appendix of REPORT.md ("Gold Label Evaluation").

Context: the course provides a shared gold-label file for 60 EStG-§-23
questions with human-written correct answers and sources. All teams have
access to this file. This is the closest approximation to an actual accuracy
measurement we can produce - unlike evaluation.py (Stage 1, §3 of REPORT.md),
which uses Model 1 as a silver / pseudo-reference, here we measure each model
against real human-written correct answers.

This is the narrowest but strictest stage of the pipeline: only 60 of the 643
questions are covered, and only the EStG §23 (Gewerbebetrieb) topic, so
results complement but do not replace the all-643 evaluations in Stage 1 and
Stage 2. Placed in the Appendix because it follows the main report.

Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU (same as evaluation.py)

Usage:
    python3 evaluation_gold.py
Output:
    ../results/evaluation_gold_table.csv        - ROUGE/BLEU per model vs gold (60 questions)
    ../results/evaluation_gold_per_question.csv - per-question ROUGE-L
"""

import os
import pandas as pd
from rouge_score import rouge_scorer
import sacrebleu

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "results"))

GOLD_CSV    = os.path.join(RESULTS_DIR, "gold_labels_EStG23.csv")

MODEL_FILES = {
    "Model1_API_Llama70B":      os.path.join(RESULTS_DIR, "model1_api_results.csv"),
    "Model2_Finetuned_Gemma2B": os.path.join(RESULTS_DIR, "model2_finetuned_results.csv"),
    "Model3_RAG_Gemma2B":       os.path.join(RESULTS_DIR, "model3_rag_results.csv"),
}


# ---------------------------------------------------------------------------
# 2. Load gold labels and model outputs, align by id
# ---------------------------------------------------------------------------
def load_data():
    gold = pd.read_csv(GOLD_CSV)
    gold.columns = [c.lstrip("\ufeff") for c in gold.columns]

    # We only care about id and correct_answer from the gold file
    gold = gold[["id", "correct_answer"]].copy()

    # Attach each model's answer for the same 60 question IDs
    for name, path in MODEL_FILES.items():
        df = pd.read_csv(path).rename(columns={"answer": name})
        gold = gold.merge(df[["id", name]], on="id", how="left")

    assert len(gold) == 60, f"expected 60 rows, got {len(gold)}"
    for name in MODEL_FILES:
        assert gold[name].notna().all(), f"{name} has missing answers for gold subset"

    return gold


# ---------------------------------------------------------------------------
# 3. Metric helpers (same as evaluation.py)
# ---------------------------------------------------------------------------
def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1) / len(r1),
        "rouge2": sum(r2) / len(r2),
        "rougeL": sum(rl) / len(rl),
    }, rl


def compute_bleu(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="intl")
    return bleu.score


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    print("Loading gold labels + model outputs...")
    df = load_data()
    print(f"Loaded {len(df)} gold-labelled questions.\n")

    gold_refs = df["correct_answer"].tolist()
    models    = list(MODEL_FILES.keys())

    rows = []
    per_q = {"id": df["id"].tolist()}

    for m in models:
        preds = df[m].tolist()
        rouge, rl_list = compute_rouge(preds, gold_refs)
        bleu            = compute_bleu(preds, gold_refs)
        rows.append({
            "model":          m,
            "rouge1_vs_gold": rouge["rouge1"],
            "rouge2_vs_gold": rouge["rouge2"],
            "rougeL_vs_gold": rouge["rougeL"],
            "bleu_vs_gold":   bleu,
        })
        per_q[f"rougeL_{m}"] = rl_list

    result_df = pd.DataFrame(rows)
    per_q_df  = pd.DataFrame(per_q)

    # Save outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_gold_table.csv"), index=False)
    per_q_df.to_csv(os.path.join(RESULTS_DIR, "evaluation_gold_per_question.csv"), index=False)

    # Pretty print
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("=== GOLD-LABEL RESULTS (60 EStG-§23 questions) ===")
    print(result_df.to_string(index=False))
    print("\nFiles written to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
