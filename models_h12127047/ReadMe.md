# ⚖️ Austrian Legal LLM: Project Report & Documentation

This project evaluates the effectiveness of fine-tuning and Retrieval-Augmented Generation (RAG) for answering questions about Austrian tax law using specialized domain knowledge (RIS database).

---

## 🏗 Model Architecture & Design

### **1. Base Model Selection**

- **Model**: `dbmdz/german-gpt2`
- **Architecture**: GPT-2 (German variant)
- **Parameters**: ~117 Million
- **Tokenizer**: Byte-level BPE optimized for German text.
- **Why this model?**: It provides a strong foundation for German language understanding, though its small size necessitates careful domain adaptation for complex legal reasoning.

### **2. Pipeline Overview**

- **Step 1: Data Acquisition (`fetchFromRIS.py`)**: Rule-based scraping of the Austrian RIS database.
- **Step 2: Domain Adaptation (`pre_train.py`)**: Knowledge injection via causal language modeling on raw legal texts.
- **Step 3: Instruction Fine-Tuning (`fine_tune.py`)**: Supervised fine-tuning (SFT) using "Question -> Answer" templates.
- **Step 4: RAG Implementation**: Injecting retrieved legal context into the model prompt during inference.

---

## 🛠 Training Methodology & Hyper-parameters

### **Domain Adaptation (Pre-training)**

- **Data**: 685 legal paragraphs from Austrian EStG, UStG, KStG, and BAO.
- **Preprocessing**: Removal of HTML artifacts, sanitization of non-printable characters.
- **Hyper-parameters**:
  - **Epochs**: 4
  - **Learning Rate**: $5 \times 10^{-6}$
  - **Training Batch Size**: 1 (with 4 Gradient Accumulation Steps)
  - **Precision**: Float32 (for numerical stability in small models)
  - **Weight Decay**: 0.05

### **Instruction Fine-Tuning**

- **Data Source**: `resources/fine_tuning.csv`
- **Method**: Supervised Fine-Tuning with prompt-response masking (only computing loss on the answer).
- **Hyper-parameters**:
  - **Epochs**: 5
  - **Learning Rate**: $2 \times 10^{-5}$
  - **Batch Size**: 4 (with 4 Gradient Accumulation Steps)
  - **Template**: `Frage: {question}\nAntwort: {answer}`

---

## 📚 RAG Pipeline: Retrieval Strategy

### **1. Retrieval Model**

The system uses a **Rule-Based RIS Retriever** implemented in `fetchFromRIS.py`.

- **Logic**: It expands common legal abbreviations (e.g., "EStG" -> "Einkommensteuergesetz") and constructs direct search URLs for the RIS database.
- **Indexing**: Live construction of queries rather than a pre-indexed vector store, ensuring access to the latest legal versions.

### **2. Preprocessing & Context Injection**

- **Chunking**: Legal documents are chunked by normative sections (Paragraphs/Articles).
- **Passages Provided**: 1 high-relevance legal paragraph is provided as the "Prime Context" for the generation model.

---

## 📊 Evaluation & Results

We evaluated three model variants against a ground truth of 685 legal QA pairs (`answers.csv`) using the **ROUGE-L** metric.

| Model Variant      | ROUGE-L Score | Description                                           |
| :----------------- | :------------ | :---------------------------------------------------- |
| **Inference Only** | **0.1136**    | Base `german-gpt2` without specialized tuning.        |
| **Fine-Tuned**     | 0.0900        | Model after domain adaptation and instruction tuning. |
| **RAG-Based**      | 0.0849        | Fine-tuned model supplemented with RIS context.       |

> [!NOTE]
> Surprisingly, the **Inference Only** model performed best in automated metrics. Below we analyze why the "advanced" methods showed divergence.

---

## 🔍 Analysis & Error Interpretation

### **The "Performance Paradox"**

The observed drop in ROUGE-L scores after fine-tuning and RAG suggests several common failure modes in legal LLM training:

1.  **"Law Hallucination"**:
    - The model often mixed up specific legal codes. For example, during inference, it cited `§ 243a UrhG` (Copyright Law) for questions regarding Corporate Tax.
2.  **Model Divergence (SFT)**:
    - Fine-tuning on a small dataset (GPT-2) caused the model to over-learn the _structure_ of legal answers (starting with "§ ...") at the expense of factual grounding.
3.  **Retrieval Noise**:
    - The current RAG implementation occasionally retrieved sections that were structurally similar but legally irrelevant, which "distracted" the small base model during generation.
4.  **Metric Bias (Sequence Length)**:
    - **Observation**: The `inference_only` model generated much longer responses than the fine-tuned/RAG versions.
    - **Impact**: Since ROUGE-L is based on the longest common subsequence, the verbosity of the base model increased the mathematical likelihood of overlapping with the ground truth, potentially inflating its score despite lower legal precision.

---

## 🚀 Future Improvements

Based on the analysis, the following steps are recommended to improve accuracy and reasoning:

- **Model Scaling**: Transitioning to a larger base model (e.g., **Llama-3-8B** or **Mistral-7B**) would significantly improve the internal knowledge base and reasoning capabilities.
- **Semantic Vector Search**: Replacing keyword-based scraping with a vector database (**FAISS/ChromaDB**) using specialized **Legal-BERT embeddings** would drastically reduce retrieval noise.
- **LoRA (Low-Rank Adaptation)**: Instead of full fine-tuning, using LoRA would allow the model to learn legal instructions without "erasing" its general-purpose language capabilities (catastrophic forgetting).
- **Precise Chunking**: Implementing recursive character splitting and metadata filtering for the RIS scraper to ensure only normative text is fed to the RAG pipeline.

---
