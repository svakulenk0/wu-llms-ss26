import pandas as pd
import os
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

TRAINING_DATA = "training_data.csv"
CLEAN_DATA = "dataset_clean.csv"
DIR_MODEL = "./finished_model"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_FILE = "faiss_index.bin"
DOCUMENTS_FILE = "documents.pkl"
OUTPUT_CSV = "rag.csv"

# ==========================================
# PART 1: BUILD RAG VECTOR DATABASE
# ==========================================
def build_database():
    if os.path.exists(FAISS_FILE): return
    contexts = pd.read_csv(TRAINING_DATA)['train'].dropna().unique().tolist()
    
    emodel = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = emodel.encode(contexts, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, FAISS_FILE)
    with open(DOCUMENTS_FILE, "wb") as f: pickle.dump(contexts, f)

# ==========================================
# PART 2: EVALUATING WITH RAG (INFERENCE)
# ==========================================
def run_eval():
    tokenizer = AutoTokenizer.from_pretrained(DIR_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DIR_MODEL)
    index = faiss.read_index(FAISS_FILE)
    with open(DOCUMENTS_FILE, "rb") as f: docs = pickle.load(f)
    emodel = SentenceTransformer(EMBEDDING_MODEL)

    if not os.path.exists(OUTPUT_CSV): pd.DataFrame(columns=["id", "answer"]).to_csv(OUTPUT_CSV, index=False)
    processed = set(pd.read_csv(OUTPUT_CSV)["id"].astype(str).tolist())
    
    for _, row in pd.read_csv(CLEAN_DATA).iterrows():
        qid, q = str(row["id"]), str(row["prompt"])
        if qid in processed: continue
        
        try:
            vec = emodel.encode([q], convert_to_numpy=True)
            dist, idxs = index.search(vec, 1)
            ctx = docs[idxs[0][0]]
            if len(ctx) > 4000: ctx = ctx[:4000] + "... [Text Truncated]"

            prompt = f"System: You are answering Austrian tax law questions.\nUser: Context: {ctx}\nQuestion: {q}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        except:
            ans = ""

        pd.DataFrame([{"id": qid, "answer": ans}]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

if __name__ == "__main__":
    build_database()
    run_eval()
