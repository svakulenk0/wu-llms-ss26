import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)

TRAINING_DATA = "training_data.csv"
CLEAN_DATA = "dataset_clean.csv"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DIR_MODEL = "./finished_model"
OUTPUT_CSV = "fine-tuning.csv"

# ==========================================
# PART 1: FINE-TUNING THE MODEL
# ==========================================
class LegalQADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"System: You are answering Austrian tax law questions.\nUser: Context: {row['train']}\nQuestion: {row['Steuerrechtliche Frage']}\nAssistant: {row['Antwort']}"
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length)
        input_ids = encoding["input_ids"]
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(encoding["attention_mask"]), "labels": torch.tensor(input_ids)}

def train_model():
    if os.path.exists(DIR_MODEL): return
    df = pd.read_csv(TRAINING_DATA)[:500]
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", num_train_epochs=3, per_device_train_batch_size=2, save_steps=100, report_to="none"),
        train_dataset=LegalQADataset(train_df, tokenizer), eval_dataset=LegalQADataset(val_df, tokenizer),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    trainer.train()
    trainer.save_model(DIR_MODEL)
    tokenizer.save_pretrained(DIR_MODEL)

# ==========================================
# PART 2: EVALUATING THE MODEL (INFERENCE)
# ==========================================
def run_eval():
    tokenizer = AutoTokenizer.from_pretrained(DIR_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DIR_MODEL)
    if not os.path.exists(OUTPUT_CSV): pd.DataFrame(columns=["id", "answer"]).to_csv(OUTPUT_CSV, index=False)
    
    processed = set(pd.read_csv(OUTPUT_CSV)["id"].astype(str).tolist())
    for _, row in pd.read_csv(CLEAN_DATA).iterrows():
        qid, q = str(row["id"]), str(row["prompt"])
        if qid in processed: continue
        
        try:
            inputs = tokenizer(f"System: You are answering Austrian tax law questions.\nUser: Question: {q}\nAssistant:", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Assistant:")[-1].strip()
        except: 
            ans = ""

        pd.DataFrame([{"id": qid, "answer": ans}]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

if __name__ == "__main__":
    train_model()
    run_eval()
