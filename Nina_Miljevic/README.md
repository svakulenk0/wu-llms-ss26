# Nina Miljevic – LLM Assignment Submission

This submission includes three different approaches for answering Austrian tax law questions:

## 1. Inference (API-based)
- Model: Groq API (LLaMA 3.1)
- Approach: Direct question answering without additional training
- File: `code/code_inference.py`
- Output: `results/results_inference_groq_clean.csv`

## 2. Fine-tuned Model
- Model: Qwen2.5-1.5B-Instruct
- Method: Supervised fine-tuning using LoRA
- Environment: Google Colab
- File: `code/code_fine-tune_google_colab.py`
- Output: `results/results_finetuned_qwen.csv`

## 3. Retrieval-Augmented Generation (RAG)
- Approach: Combines retrieval of legal text with LLM generation
- Components:
  - Embeddings: SentenceTransformers (multilingual MiniLM)
  - Vector store: FAISS
  - Generator: Groq API (LLaMA 3.1)
- File: `code/code_rag.py`
- Output: `results/results_rag_groq.csv`

### Legal Sources Used for RAG
The following Austrian legal sources were used:
- Einkommensteuergesetz (EStG)
- Körperschaftsteuergesetz (KStG)
- Umsatzsteuergesetz (UStG)
- Bundesabgabenordnung (BAO)
- Unternehmensgesetzbuch (UGB)
- Grunderwerbsteuergesetz (GrEStG)

These were processed locally as PDF files, converted into text, chunked, and embedded for retrieval.

Due to file size and copyright considerations, the original PDFs are not included in this repository.

## Reproducibility
All code required to reproduce the results is included in the `code` folder.

The test dataset was provided by the course and is not modified.

## Notes
- Each result file contains answers for all dataset questions.
- Answers were constrained to be concise and formatted as single-line outputs.