# Austrian Tax Law AI - Project Report

## 1. Models and Configuration Setup

For this project, three distinct approaches were evaluated to automate the processing of Austrian legal tax questions. 

**Models Used:**
1. **API Baseline (Inference):** `Llama-3-70b` accessed via the Groq Cloud API.
2. **Local Fine-Tuned Model:** `Qwen/Qwen2.5-0.5B` (a 0.5 billion parameter causal language model).
3. **Local RAG-Enhanced Model:** `Qwen/Qwen2.5-0.5B` augmented with FAISS vector search.

**Pre-Training Data:**
The `Qwen2.5-0.5B` model was pre-trained by the Qwen team on large-scale, high-quality, multilingual datasets encompassing up to 18 trillion tokens (including extensive code, mathematics, and multilingual web text). 

**Sampling Approach:**
During inference across all local models, a standard probabilistic sampling approach was utilized with the following hyper-parameters:
- `temperature`: 0.7 (allowing for natural language variation while remaining grounded in facts)
- `do_sample`: True
- `max_new_tokens`: 150
- `pad_token_id`: EOS token

---

## 2. Fine-Tuning Specification

To adapt the base `Qwen2.5-0.5B` model to specific Austrian tax law phrasing, supervised fine-tuning was performed.

* **Data used:** `training_data.csv` containing specific input-output pairs of Austrian Tax legislation and corresponding responses.
* **How it was fine-tuned:** A custom PyTorch `Dataset` class (`LegalQADataset`) was created to concatenate the Prompt, Context, and Assistant Answers into a single causal modeling sequence. The HuggingFace `Trainer` class was used to perform standard end-to-end fine-tuning on the weights.
* **Hyper-Parameters:**
  - **Epochs:** 3
  - **Batch Size:** 2 (`per_device_train_batch_size`)
  - **Max Sequence Length:** 256 tokens (with padding and truncation)
  - **Loss Function Strategy:** Standard Causal Language Modeling over the tokenized texts.

---

## 3. Retrieval-Augmented Generation (RAG) Architecture

In place of fine-tuning, the third model utilized an external knowledge base retrieval system without modifying the base model weights.

* **Retrieval Model:** We utilized the `paraphrase-multilingual-MiniLM-L12-v2` SentenceTransformer. This model is specifically optimized for multilingual semantic similarity mapping (essential for German legal text).
* **Documents Indexed:** We extracted the unique legal contexts directly from the `train` column of the `training_data.csv`.
* **Preprocessing & Chunking:** The provided legal contexts were embedded as whole passages (row-based). However, during inference, a dynamic chunking/truncation approach was used: contexts were strictly truncated at 4,000 characters to prevent overflowing the memory limits of the local environment.
* **Passages Given:** `k = 1` (The single most mathematically similar context was retrieved using an L2 distance search on a `FAISS` database and injected into the system prompt).

---

## 4. Evaluation and Final Results

Model performance was automatically evaluated against the ground-truth answers in the `Austrian Tax Law Dataset` utilizing three deterministic NLP metrics:
* **BERTScore (F1):** For measuring semantic meaning and conceptual accuracy.
* **ROUGE (1, 2, L):** For measuring N-gram/word inclusion overlap.
* **BLEU:** For strict exact-match sequence precision.

Our automated `evaluate_models.py` script deduplicated and evaluated the overlapping `562` questions.

### Main Result Table

| Model | Evaluated Questions | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 |
|:---|---:|---:|---:|---:|---:|---:|
| **Groq API Baseline** | 562 | 21.96 | 6.80 | 14.61 | 3.64 | **69.87** |
| **Fine-Tuned Local** | 562 | 14.53 | 3.76 | 10.17 | 2.27 | **64.25** |
| **RAG Enhanced Local**| 562 | 11.24 | 2.61 | 7.87 | 1.71 | **62.74** |

**Which model performed best?**
The **Groq API Baseline (Llama 3 70B)** strongly outperformed the local models, achieving the highest semantic accuracy (BERTScore: ~70) and phrase overlap (ROUGE-L: ~14.6). Among the local 0.5B models, the **Fine-Tuned** approach outperformed the RAG approach. 

---

## 5. Bonus: Error Analysis 

Our evaluation metrics highlight a notorious challenge in generative legal AI. The BLEU scores are extremely low (< 4) universally, yet the BERTScores are high (> 60). This indicates successful context capture, but varied delivery.

**Main Issues and Model Mistakes:**

1. **Groq API (The "Wordy" Hallucinator):** 
   - *Mistake:* Groq generated highly conversational, long-winded answers. The ground truth in Austrian tax law is often strict (e.g., merely stating *"§ 7 Abs. 1 KStG 1988"*). Groq would respond, *"According to the Austrian tax law, the specific article you are looking for is..."* which heavily penalized its ROUGE and BLEU scores, despite being conceptually correct.

2. **Fine-Tuned Local (The "Over-Fitted" Robot):** 
   - *Mistake:* While fine-tuning taught the 0.5B model to mimic the strict, blunt formatting of the CSV, it suffered from limited deductive logic. When asked a nuanced question that wasn't perfectly represented in its training batches, it would "guess" law section numbers by combining unrelated clauses it had memorized.

3. **RAG Local (The "Distracted" Reader):**
   - *Mistake:* RAG performed the worst overall. Because we relied on a `k=1` retrieval limit and a hard `4000` character truncation constraint to protect local memory limits, the model occasionally received the wrong paragraph. If two tax codes possessed similar wording, the `FAISS` index would fetch the wrong one, causing the generative model to confidently output irrelevant facts. 
   - *Cross-Model Consistency:* Unlike Groq which failed due to being conversational, RAG failed purely due to "Context Missing" errors. If the answer wasn't in the retrieved chunk, the 0.5B model lacked the inherent world-knowledge to guess correctly.
