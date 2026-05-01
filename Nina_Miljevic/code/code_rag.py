from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
PDF_DIR = BASE_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "nina_rag_groq.csv"

# -----------------------------
# 2. Load test dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df))


# -----------------------------
# 3. Extract text from PDFs
# -----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

documents = []

for pdf_file in PDF_DIR.glob("*.pdf"):
    print(f"Processing {pdf_file.name}")
    text = extract_text_from_pdf(pdf_file)

    documents.append({
        "source": pdf_file.name,
        "text": text
    })

# -----------------------------
# 4. Chunking
# -----------------------------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 100:
            chunks.append(chunk)
    return chunks

chunks = []

for doc in documents:
    doc_chunks = chunk_text(doc["text"])

    for chunk in doc_chunks:
        chunks.append({
            "source": doc["source"],
            "text": chunk
        })

print(f"Total chunks: {len(chunks)}")

# -----------------------------
# 5. Embeddings
# -----------------------------
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

texts = [c["text"] for c in chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True)

# -----------------------------
# 6. FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# 7. Groq setup
# -----------------------------
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("Set GROQ_API_KEY first!")

client = Groq(api_key=api_key)

# -----------------------------
# 8. Retrieval function
# -----------------------------
def retrieve_context(question, k=3):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunks[idx]["text"])

    return "\n\n".join(retrieved)

# -----------------------------
# 9. Generate answer
# -----------------------------
def generate_answer(question, context):
    prompt = f"""Du bist ein Experte für österreichisches Steuerrecht.

Nutze ausschließlich die folgenden rechtlichen Grundlagen zur Beantwortung.

Kontext:
{context}

Frage: {question}

Antworte präzise auf Deutsch in höchstens 2 Sätzen.
Keine Aufzählungen, keine Wiederholung der Frage.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_completion_tokens=150,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    # clean formatting
    answer = answer.replace("\n", " ").replace("\r", " ")
    answer = " ".join(answer.split())

    return answer

# -----------------------------
# 10. Run RAG
# -----------------------------
results = []

for i, row in df.iterrows():
    question_id = row["id"]
    question = str(row["prompt"])

    context = retrieve_context(question)

    try:
        answer = generate_answer(question, context)
    except Exception as e:
        answer = f"ERROR: {e}"

    print("\n---")
    print(f"Question {question_id}")
    print(f"Answer: {answer}")

    results.append({
        "id": question_id,
        "answer": answer
    })

# -----------------------------
# 11. Save CSV
# -----------------------------
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\nDone!")
