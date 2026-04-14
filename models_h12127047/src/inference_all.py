# --- 1. Installation ---
!pip install transformers datasets accelerate pandas scikit-learn -U

import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# --- 2. Pfade definieren ---
input_model_path = "./my_legal_model"  # Dein Start-Modell
ft_file = "fine_tuning.csv"            # Deine neuen Trainingsdaten
output_path = "./legal_model_final"    # Hier landet das verbesserte Modell

# --- 3. Daten laden ---
try:
    # Robustes Laden der CSV
    df = pd.read_csv(ft_file, sep=None, engine='python', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    
    # Sicherstellen, dass die Spalte 'train' existiert
    if 'train' not in df.columns:
        potential = [c for c in df.columns if 'train' in c.lower()]
        target = potential[0] if potential else df.columns[1]
        df = df.rename(columns={target: 'train'})
    
    df = df.dropna(subset=['train'])
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"✅ Daten bereit: {len(train_df)} Training-Zeilen.")
except Exception as e:
    print(f"❌ Fehler beim Laden der CSV: {e}")
    raise e

# --- 4. Dataset Klasse ---
class LegalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question = str(row.get('Full Reference', 'Steuerrechtliche Frage'))
        answer = str(row['train'])
        
        # Format: Frage -> Antwort
        full_text = f"Frage: {question}\nAntwort: {answer}{self.tokenizer.eos_token}"
        
        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length")
        labels = list(enc["input_ids"])
        
        # Maskierung des Prompts für den Loss
        prompt_text = f"Frage: {question}\nAntwort: "
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        
        for i in range(len(labels)):
            if i < prompt_len or enc["input_ids"][i] == self.tokenizer.pad_token_id:
                labels[i] = -100
                
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# --- 5. Modell & Tokenizer laden (aus my_legal_model) ---
print(f"Lade existierendes Modell von {input_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(input_model_path)
# Falls kein Pad-Token gesetzt ist (wichtig bei GPT-Modellen)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(input_model_path).to("cuda")
model.resize_token_embeddings(len(tokenizer))

# --- 6. Training Konfiguration ---
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=5,            # Da das Modell schon Vorwissen hat, reichen oft 5 Epochen
    learning_rate=2e-5,            # Etwas niedrigere LR für Fine-Tuning eines bestehenden Modells
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="no",
    save_total_limit=1,
    fp16=True,                     
    report_to="none"
)

# --- 7. Trainer Start ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=LegalDataset(train_df, tokenizer),
)

print("🚀 Starte Fine-Tuning auf Basis von 'my_legal_model'...")
trainer.train()

# --- 8. Speichern ---
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ Abgeschlossen! Dein verbessertes Modell liegt in: {output_path}")