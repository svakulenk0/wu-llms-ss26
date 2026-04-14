This README provides a comprehensive guide to your legal AI pipeline, specifically optimized for **Google Colab**. This pipeline allows you to fetch Austrian legal texts from the RIS (Rechtsinformationssystem) database and perform a two-stage training process (Pre-training and Fine-tuning) using a German GPT-2 base model.

Since Google Colab environment was chosen, file-paths may be different from local setups and it is strongly advised to double-check them.

---

# Legal AI Training Pipeline: Austrian Law

This repository contains a suite of Python scripts designed to automate the creation of a specialized German legal language model. It handles everything from data acquisition via web scraping to model refinement.

## 📋 System Requirements
* **Environment:** [Google Colab](https://colab.research.google.com/) (Recommended: T4 or L4 GPU Runtime).
* **Base Model:** `dbmdz/german-gpt2`
* **Language:** German (Input/Output).

---

## 🚀 Step-by-Step Guide

### Step 1: Data Acquisition (`fetchFromRIS.py`)
This script takes a list of legal references (e.g., "§ 1 EStG") and fetches the corresponding full text from the official Austrian RIS database.

1.  **Input:** Create an input CSV file containing legal references separated by semicolons.
2.  **Execution:**
    ```bash
    python fetchFromRIS.py input_references.csv (Google csv sheet, with all paragraphs) training_data.csv
    ```
3.  **Functionality:** It expands abbreviations (e.g., "EStG" to "Einkommensteuergesetz"), constructs a search URL, finds the document on RIS, and extracts the relevant paragraph text.

### Step 2: Base Training / Domain Adaptation (`pre_train.py`)
This stage teaches the `german-gpt2` model the specific language and structure of Austrian law.

1.  **Setup:** Ensure `training_data.csv` (from Step 1) is in your Colab file folder.
2.  **Process:**
    * Cleans non-standard characters to prevent tokenizer crashes.
    * Masks the prompts so the model only learns to generate the legal content (`Inhalt`).
    * Uses a hyper-stable configuration (Float32, low learning rate) to avoid "NaN" errors.
3.  **Output:** Saves the adapted model to `./my_legal_model`.

### Step 3: Instruction Fine-Tuning (`fine_tune.py`)
This script refines the model to answer specific questions or follow instruction formats using your own Q&A data.

1.  **Input:** Requires a file named `fine_tuning.csv` with a column containing your Q&A pairs (the script targets a column named `train` or similar).
2.  **Process:** It loads the model created in Step 2 and performs 5 epochs of high-intensity learning.
3.  **Output:** Saves the final production-ready model to `./legal_model_final`.

### Step 4: Batch Inference (`inference_all.py`)
Use this script to generate answers for a large set of questions using your final model.

1.  **Functionality:** It loads the model from `./legal_model_final` and processes a CSV file to generate legal answers in bulk.
2.  **Format:** It uses a "Question -> Answer" prompt template to ensure the model remains in "legal advisor" mode.

---

## 🛠 Script Overview

| File | Purpose | Key Feature |
| :--- | :--- | :--- |
| `fetchFromRIS.py` | Data Scraping | Automatic expansion of law abbreviations (EStG, BAO, etc.). |
| `pre_train.py` | Knowledge Injection | Specialized sanitization for the German GPT-2 tokenizer. |
| `fine_tune.py` | Instruction Tuning | Loss masking (the model isn't graded on the question, only the answer). |
| `inference_all.py` | Batch Prediction | Optimized for CUDA (GPU) acceleration. |

---

## ⚠️ Important Notes for Google Colab

1.  **GPU Acceleration:** Before running `pre_train.py` or `fine_tune.py`, go to `Runtime -> Change runtime type` and select **T4 GPU**.
2.  **Library Dependencies:** All scripts include a `pip install` command at the top. Ensure these run successfully to install `transformers`, `peft`, and `accelerate`.
3.  **Character Encoding:** Legal texts often contain special characters (like `§` or non-breaking spaces). The scripts use `.encode("utf-8", errors="ignore")` to ensure the training doesn't crash on "Index out of range" errors.
4.  **File Cleanup:** `pre_train.py` contains a "Nuclear Cleanup" section that deletes old results. Be sure to back up your models if you intend to keep multiple versions.