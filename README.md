# ==============================================================================
# PROJECT REPORT: Austrian Tax Law LLM
# ==============================================================================

## 1. Models and Baseline Configuration
### Initial Approach (Failure Mode)
At the beginning, I used the `google/flan-t5-small` architecture across all of my models. However, this model proved to be far too small. It hallucinated in all models and produced a mixture of English and German outputs that didn't make any sense at all. 

### Final Stable Configurations
Because of this, I switched my approach and used the much stronger `Qwen2.5` models instead, which finally produced excellent, high-quality German outputs:
- **Model 1 (Zero-Shot Inference):** `Qwen/Qwen2.5-3B-Instruct`
- **Model 2 (Fine-Tuning):** `Qwen/Qwen2.5-1.5B-Instruct`
- **Model 3 (RAG):** `Qwen/Qwen2.5-3B-Instruct` (Generator) + `MiniLM-L12` (Retriever)

### Hyper-parameters & Setup (Model 1 Baseline):
- **Model Size:** 3 Billion parameters.
- **Sampling Approach:** Stochastic generation with low temperature (`T=0.1`) and `do_sample=True` to allow slight variability while maintaining high precision. `max_new_tokens` was set to 250.
- **Pre-training Data:** Qwen2.5 is pre-trained on a massive multilingual corpus (up to 18T tokens) encompassing coding, math, and strong multi-lingual logic, which explains its native fluency in German tax logic compared to FLAN-T5.

---

## 2. Fine-Tuned Model (Model 2)
### Data
Since a pre-existing instruction dataset for Austrian Tax Law was absent, an **automated self-instruct pipeline** was built to generate training data:
- **TF-IDF Retrieval:** For each of the 644 questions, the most relevant PDF paragraph was retrieved using a TF-IDF vectorizer (max_features=10,000, filtered for common German stop words). 
- **Training Pairs:** 150 high-quality question-context pairs were created. The model was trained to answer the question strictly based on the provided context (Causal LM objective using ChatML format).

### Fine-Tuning Strategy
- **Method:** PEFT / LoRA (Low-Rank Adaptation) was used to train the 1.5B parameter model within the Kaggle T4 (15GB VRAM) constraints.
- **Quantization:** 4-bit NormalFloat (NF4) quantization via BitsAndBytes.
- **Hyper-parameters:**
  - Rank (r) = 8, Alpha = 16, Dropout = 0.05
  - Target Modules: `q_proj`, `v_proj`
  - Max Steps = 50, Batch Size = 2, Gradient Accumulation = 4
  - Optimizer: AdamW, Learning Rate = 2e-4, FP16 precision.

---

## 3. RAG-Based Model (Model 3)
### Retrieval Model
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (a dense neural vector model highly optimized for multilingual semantic search).

### Document Indexing & Preprocessing (Chunking)
- The official Tax Law PDFs were loaded utilizing `pypdf`.
- **Chunking:** Pages were extracted, stripped of noise (e.g., footers/emails), and split by double line breaks (`\n\n`) into semantic paragraphs. Small fragments (< 80 characters) were discarded.
- **Embedding:** All cleaned paragraphs were encoded into dense vectors using L2 normalization to allow rapid Cosine Similarity matching (via dot product).

### Input Passages (Top-K)
- For every question, the generator received the **top 3 (k=3)** most semantically similar paragraphs. The maximum length of the concatenated retrieved context was capped at 1000 characters to prevent context-window overflow.

---

## 4. Evaluation & Performance Matrix
Since the shared annotation task was not finalized, the full official `Austrian Tax Law Dataset` comprising roughly 680 queries paired with their `correct_answer` was utilized as the Ground Truth baseline. 

An automated pipeline (`evaluate_models.py`) iterated over the entire dataset, comparing the LLM-generated output against the provided human-curated ground truth to calculate the BLEU (SacreBLEU) and ROUGE-L metrics.

### Main Result Table
| Model Strategy | Base Model | BLEU Score | ROUGE-L (F1) | BERTScore | Execution Setup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1 (Zero-Shot)** | `Qwen2.5-3B-Instruct` | **3.13** | **0.1492** | **0.6975** | Prompt Only |
| **Model 2 (Fine-Tuned)** | `Qwen2.5-1.5B-Instruct` | 1.31 | 0.0801 | 0.6537 | LoRA (r=8) |
| **Model 3 (RAG)** | `Qwen2.5-3B-Instruct` | 2.72 | 0.1264 | 0.6792 | Top-3 Semantic |

**Conclusion:** Generative LLMs naturally paraphrase information rather than outputting exact memorized string sequences. As a result, the n-gram based metrics (BLEU and ROUGE-L) severely penalized all models. However, the newly integrated **BERTScore**—which evaluates mathematical semantic distance via high-dimensional embeddings—painted a much more accurate picture: all models achieved a semantic similarity of roughly ~0.65 to 0.70 to the human ground truth. Once again, the sheer baseline power of the `Qwen2.5-3B` parameter model (Model 1) marginally outcompeted the others, though the RAG Model (Model 3) followed very closely by dynamically fetching evidence.

---

## 5. Bonus: Error Analysis
Throughout development, several distinct failure modes were identified across the models:

1. **Model 1 (Hallucination without grounding):**
   - *Mistake:* Pure zero-shot models often confidently asserted generic German tax logic (e.g., German BGB/EStG) instead of Austrian specificities (öKStG/öEStG) due to the overwhelming presence of federal German data in their pre-training corpus.
2. **Model 2 (Metadata Leakage & Infinite Loops):**
   - *Mistake:* During initial tests, the fine-tuned model outputted strings like `"Lizenziert for: Viktoria.Schwab@aau.at"`. It had memorized the noisy footers of the training PDFs.
   - *Fix:* Implementing strict regex pipeline filtering prior to tokenization resolved this. Furthermore, the `flan-t5-small` models I originally used frequently fell into massive infinite repetition loops. Because the models hallucinated infinitely without ever hitting a stop token, extracting the dataset metrics on Kaggle required me to do 3 separate agonizing runs, each taking between 4 to 7 hours, only to output completely unusable text. By switching to `Qwen2.5` and enforcing strict generation penalties (`no_repeat_ngram_size=3`, `repetition_penalty=1.2`), I resolved the generative looping entirely. The final pipeline now runs fast and the output is finally highly accurate and good.
3. **Model 3 (Retrieval Misses):**
   - *Mistake:* If the question used highly abstracted phrasing not semantically aligned with the dense legal text, the MiniLM retriever fetched irrelevant paragraphs. When given irrelevant context, the generator either refused to answer or forced a hallucination.
   - *Fix:* Implementing a strict system prompt ("NUTZE AUSSCHLIESSLICH den bereitgestellten Kontext") heavily reduced RAG-hallucinations, translating retrieval misses into safe "I don't know" abstentions.
