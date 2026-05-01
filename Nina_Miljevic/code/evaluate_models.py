from pathlib import Path
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

# -----------------------------
# 1. Set Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # goes from code/ up to Nina_Miljevic/

GOLD_PATH = BASE_DIR / "data" / "austrian_tax_law_dataset.csv"
INF_PATH = BASE_DIR / "results" / "results_inference_groq_clean.csv"
FIN_PATH = BASE_DIR / "results" / "results_finetuned_qwen.csv"
RAG_PATH = BASE_DIR / "results" / "results_rag_groq.csv"

OUTPUT_EVAL_PATH = BASE_DIR / "results" / "evaluation_results.csv"
OUTPUT_ERRORS_PATH = BASE_DIR / "results" / "error_analysis_sample.csv"

# -----------------------------

def clean(text):
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()

def load_gold(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip()
    df = df[["id", "correct_answer"]].copy()
    df["correct_answer"] = df["correct_answer"].apply(clean)
    return df

def load_prediction(path, model_name):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # expected format: id, answer
    if "id" not in df.columns:
        raise ValueError(f"'id' column not found in {path}")

    if "answer" not in df.columns:
        # if the second column is not literally called answer, use the second column
        if len(df.columns) >= 2:
            second_col = df.columns[1]
            df = df.rename(columns={second_col: "answer"})
        else:
            raise ValueError(f"No answer column found in {path}")

    df = df[["id", "answer"]].copy()
    df["answer"] = df["answer"].apply(clean)
    df = df.rename(columns={"answer": model_name})
    return df

def bleu_score_one(ref, hyp):
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth)

def rouge_l_one(ref, hyp, scorer):
    return scorer.score(ref, hyp)["rougeL"].fmeasure


# 2. Load data
# -----------------------------
gold = load_gold(GOLD_PATH)
inf = load_prediction(INF_PATH, "inference")
fin = load_prediction(FIN_PATH, "finetuned")
rag = load_prediction(RAG_PATH, "rag")


# 3. Merge the data by id

df = gold.merge(inf, on="id", how="inner")
df = df.merge(fin, on="id", how="inner")
df = df.merge(rag, on="id", how="inner")

print("Merged rows:", len(df))
print(df.head())


# 4. Evaluate

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

results = []
error_tables = []

for model_col in ["inference", "finetuned", "rag"]:
    bleu_scores = []
    rouge_scores = []

    refs = df["correct_answer"].tolist()
    hyps = df[model_col].tolist()

    for ref, hyp in zip(refs, hyps):
        bleu_scores.append(bleu_score_one(ref, hyp))
        rouge_scores.append(rouge_l_one(ref, hyp, scorer))

    # BERTScore for all examples at once
    P, R, F1 = score(hyps, refs, lang="de", verbose=True)

    model_result = pd.DataFrame({
        "id": df["id"],
        "correct_answer": refs,
        "prediction": hyps,
        "BLEU": bleu_scores,
        "ROUGE-L": rouge_scores,
        "BERTScore_F1": F1.tolist()
    })

    model_result["model"] = model_col
    error_tables.append(model_result.sort_values("BERTScore_F1").head(10))

    results.append({
        "model": model_col,
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores),
        "BERTScore": F1.mean().item()
    })

# -----------------------------
# 5. Create and Save results table and error analysis
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_EVAL_PATH, index=False, encoding="utf-8")

error_df = pd.concat(error_tables, ignore_index=True)
error_df.to_csv(OUTPUT_ERRORS_PATH, index=False, encoding="utf-8")

print("\nMain results:")
print(results_df)

print(f"\nSaved: {OUTPUT_EVAL_PATH}")
print(f"Saved: {OUTPUT_ERRORS_PATH}")