from pathlib import Path
import os
import pandas as pd
from groq import Groq

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "nina_inference_groq.csv"

# -----------------------------
# 2. Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# -----------------------------
# 3. API Key
# -----------------------------
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

client = Groq(api_key=api_key)

# -----------------------------
# 4. Function
# -----------------------------
def generate_answer(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_completion_tokens=120,
        messages=[
            {
                "role": "system",
                "content": "Du bist Experte für österreichisches Steuerrecht. Antworte präzise auf Deutsch in 1–2 Sätzen."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content.strip()

# -----------------------------
# 5. Loop
# -----------------------------
results = []

for i, row in df.iterrows():
    question_id = row["id"]
    question = str(row["prompt"])

    try:
        answer = generate_answer(question)
    except Exception as e:
        answer = f"ERROR: {e}"

    print("\n---")
    print(f"Question {question_id}: {question}")
    print(f"Answer: {answer}")

    results.append({
        "id": question_id,
        "answer": answer
    })

# -----------------------------
# 6. Save CSV
# -----------------------------
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\nDone!")

df = pd.read_csv("results/nina_inference_groq.csv")

df["answer"] = (
    df["answer"]
    .astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df.to_csv("results/nina_inference_groq_clean.csv", index=False, encoding="utf-8")
print("Cleaned file saved.")
print("Rows:", len(df))
