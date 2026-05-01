import pandas as pd
import time
import os
from openai import OpenAI

# -----------------------------
# 1. Settings
# -----------------------------
INPUT_FILE = "dataset_clean.csv"
OUTPUT_FILE = "inference_groq.csv"

ID_COLUMN = "id"
QUESTION_COLUMN = "prompt"

MODEL_NAME = "llama-3.1-8b-instant"

client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

# Create an empty file with headers if it does not exist yet
if not os.path.exists(OUTPUT_FILE):
    pd.DataFrame(columns=["id", "answer"]).to_csv(OUTPUT_FILE, index=False)

# We load already processed answers so we don't process them again on restarts
processed_df = pd.read_csv(OUTPUT_FILE)
processed_ids = set(processed_df["id"].astype(str).tolist())

df = pd.read_csv(INPUT_FILE)
print("Columns in dataset:", df.columns.tolist())
print(f"Total rows in dataset: {len(df)}")
print(f"Already processed: {len(processed_ids)}")

def ask_model(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are answering Austrian tax law questions. Answer as accurately and concisely as possible. If unsure, give the best grounded answer."},
            {"role": "user", "content": f"Question: {question}"}
        ],
        temperature=0.7,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Process all questions
# -----------------------------
success_count = 0

for i, row in df.iterrows():
    qid = str(row[ID_COLUMN])
    question = str(row[QUESTION_COLUMN])
    
    # Skip if this question has already been saved!
    if qid in processed_ids:
        continue
        
    try:
        answer = ask_model(question)
        status = "OK"
        success_count += 1
    except Exception as e:
        err_msg = str(e).split('\n')[0][:50]
        print(f"Error on {qid}: {err_msg}...")
        answer = ""
    
    # SAVE TO FILE IMMEDIATELY:
    # Append one row to the end of the existing file (mode='a')
    new_row = pd.DataFrame([{"id": qid, "answer": answer}])
    new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    # 5-second delay so we don't hit the 6000 TPM limit on Groq
    time.sleep(5)
    
    if success_count > 0 and success_count % 5 == 0:
        print(f"Successfully processed new: {success_count}...")

print(f"Process completed! Answers in file: {OUTPUT_FILE}")