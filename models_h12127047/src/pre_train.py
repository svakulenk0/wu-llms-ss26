import pandas as pd
import torch
import os
import shutil
import sys
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# --- 0. Nuclear Cleanup ---
for folder in ["./results", "./my_legal_model"]:
    if os.path.exists(folder):
        print(f"Cleaning up {folder}...")
        shutil.rmtree(folder)

# --- 1. Load Data ---
try:
    df = pd.read_csv("training_data.csv", sep=';')
except Exception:
    df = pd.read_csv("training_data.csv")

df = df.dropna(subset=['train'])
df = df[df['train'].str.strip() != ""]
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# --- 2. Sanitized Dataset Class ---
class LegalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # SANITIZATION: Clean characters that crash the German GPT-2 tokenizer
        def clean_text(t):
            t = str(t).replace('\xa0', ' ') # Remove non-breaking spaces
            t = t.replace('\r', '')         # Remove carriage returns
            # Keep only standard printable characters to avoid 'Index out of range'
            return t.encode("utf-8", errors="ignore").decode("utf-8")

        prompt = f"Gesetzestext zu: {clean_text(row['Full Reference'])}\nInhalt:\n"
        content = clean_text(row['train']) + self.tokenizer.eos_token
        
        # Encode
        full_enc = self.tokenizer(
            prompt + content,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        
        input_ids = full_enc["input_ids"]
        labels = list(input_ids)
        
        # Determine prompt length to mask it
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # MASKING: Set prompt and padding to -100 so model only learns 'Inhalt'
        for i in range(len(labels)):
            if i < prompt_len or input_ids[i] == self.tokenizer.pad_token_id:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# --- 3. Forced Stable Model Loading ---
model_name = "dbmdz/german-gpt2" 
print(f"Loading German model: {model_name}...")

# Force the tokenizer to the original stable vocab size (50265)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if len(tokenizer) > 50265:
    print(f"Resetting tokenizer vocab from {len(tokenizer)} to 50265...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, vocab_size=50265)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.eos_token_id

# Force Float32 for CPU/MPS stability (prevents NaNs)
model = model.float()

# --- 4. Hyper-Stable Training Arguments ---
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    learning_rate=5e-6,          # Extremely low for stability
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=5,
    warmup_steps=100,
    weight_decay=0.05,
    max_grad_norm=0.3,           # Strict clipping to kill NaNs
    fp16=False,                  # DO NOT change to True on CPU/Mac
    report_to="none",
    logging_first_step=True
)

def custom_collator(features):
    return {k: torch.stack([f[k] for f in features]) for k in features[0]}

# --- 5. Run ---
print(f"\nFinal Check: Vocab Size = {len(tokenizer)} | Model Size = {model.get_input_embeddings().weight.shape[0]}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=LegalDataset(train_df, tokenizer),
    eval_dataset=LegalDataset(val_df, tokenizer),
    data_collator=custom_collator,
)

print("Starting training...")
try:
    trainer.train()
    trainer.save_model("./my_legal_model")
    tokenizer.save_pretrained("./my_legal_model")
    print("\nTraining complete. Model saved to ./my_legal_model")
except Exception as e:
    print(f"\nFATAL ERROR: {e}")