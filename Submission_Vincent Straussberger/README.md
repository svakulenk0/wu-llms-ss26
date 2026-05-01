# Vincent Straussberger Project Report

## Project Overview

This submission compares three approaches for answering Austrian corporate tax questions:

- direct inference with a public Qwen model
- LoRA fine-tuning of the same base model on a small RIS-based training set
- retrieval-augmented generation with RIS source material

All three runs were evaluated against `evaluation/solutions.csv` with BERTScore. The main comparison metric is mean BERTScore F1.

## Submission Contents

- `code/Inference_Qwen3.ipynb`
- `code/FineTune_Qwen3.ipynb`
- `code/RAG_Qwen3.ipynb`
- `code/requirements.txt`
- `code/ris_tax_training_100.csv`
- `code/ris_seed_urls.txt`
- `code/dataset_clean.csv`
- `results/inference_qwen3_0_6b.csv`
- `results/finetune_qwen3_0_6b.csv`
- `results/rag_qwen3_0_6b.csv`
- `evaluation/evaluate_bertscore.ipynb`
- `evaluation/solutions.csv`
- `evaluation/bertscore_summary.csv`
- `evaluation/bertscore_details.csv`
- `README.md`

## Model 1: Inference

- Model: `Qwen/Qwen3-0.6B`
- Parameters: approximately `0.6B`
- Task setup: direct question answering on the benchmark without additional training or retrieval
- Generation settings:
  - `max_input_tokens=420`
  - `max_new_tokens=160`
  - `temperature=0.7`
  - `top_p=0.8`
  - `do_sample=True`
- Pre-training note: this run uses the public pre-trained Qwen checkpoint released by the model provider. No additional project-specific pre-training was performed, and the original large-scale pre-training corpus was not recreated in this project.

## Model 2: Fine-Tune

- Base model: `Qwen/Qwen3-0.6B`
- Parameters: approximately `0.6B` before LoRA fine-tuning
- Fine-tuning data: `ris_tax_training_100.csv`
- Fine-tuning method: LoRA-based supervised fine-tuning
- Current local run settings:
  - `ULTRAFAST_VALIDATION_MODE=True`
  - `MAX_TRAIN_ROWS=12`
  - `MAX_SEQ_LENGTH=128`
  - `NUM_TRAIN_EPOCHS=1`
  - `PER_DEVICE_TRAIN_BATCH_SIZE=1`
  - `GRADIENT_ACCUMULATION_STEPS=1`
  - `LORA_R=2`
- Generation settings:
  - `max_input_tokens=256`
  - `max_new_tokens=64`
  - `temperature=0.7`
  - `top_p=0.8`
  - `do_sample=True`
- Pre-training note: the fine-tuned run starts from the same public pre-trained Qwen checkpoint released by the model provider. No additional project-specific pre-training was performed before fine-tuning.

## Model 3: RAG

- Generator model: `Qwen/Qwen3-0.6B`
- Generator parameters: approximately `0.6B`
- Retrieval encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Retrieval sources: RIS URLs listed in `ris_seed_urls.txt`
- Preprocessing and chunking:
  - `chunk_size=420`
  - `chunk_overlap=64`
- Retrieval settings:
  - `top_k=2`
  - `dense_weight=0.78`
  - `lexical_weight=0.22`
  - `2` retrieved passages are passed to the generator
- Generation settings:
  - `max_input_tokens=320`
  - `max_new_tokens=48`
  - `temperature=0.1`
  - `top_p=0.9`
  - `do_sample=True`
- Pre-training note: the response generator uses the same public pre-trained Qwen checkpoint released by the model provider. No additional project-specific pre-training was performed for the RAG setup.

## Evaluation Setup

- Reference file: `evaluation/solutions.csv`
- Metric: **BERTScore only**
- Reference text column: `correct_answer`
- Metric implementation: `evaluate` + `bert-base-multilingual-cased`
- Main comparison metric: mean **BERTScore F1**
- Evaluation subset: `641` matched benchmark questions
- Note: `evaluation/solutions.csv` contains one duplicated ID (`VAT-INTL-001`) and does not align perfectly with the prediction files (`VAT-INTL-081` and `VAT-INTL-082` are missing from the reference file, while `ESTG27-015` and `ESTG27-016` are reference-only). The evaluation notebook keeps the first duplicate occurrence and evaluates on the shared ID overlap.

## Main Results

| Model | BERTScore Precision | BERTScore Recall | BERTScore F1 |
| --- | ---: | ---: | ---: |
| Inference_Qwen3 | 0.724733 | 0.673400 | 0.696812 |
| FineTune_Qwen3 | 0.699984 | 0.675905 | 0.686911 |
| RAG_Qwen3 | 0.689378 | 0.662398 | 0.674974 |

## Best Model

The best model in this setup is **Inference_Qwen3**, because it achieved the highest mean BERTScore F1 (`0.696812`).

## Error Analysis

The lowest-scoring examples in `evaluation/bertscore_details.csv` show a few clear patterns:

- **Inference** performs best overall, but still fails badly when the task requires a concrete jurisdiction or location-specific conclusion. In the lowest-scoring examples it sometimes gives a very short but wrong answer such as just `"In Wien."`, or a generic explanation that does not address the exact VAT rule.
- **Fine-Tune** appears less stable than the base inference model. Some low-scoring outputs drift into generic legal/business wording, contain off-topic content, or over-explain without matching the actual reference answer closely. This is likely related to the lightweight CPU-friendly fine-tuning setup with only a very small effective training run.
- **RAG** makes the most obvious grounding mistakes. The weakest answers often import legal text from the wrong topic entirely, for example returning an interest-limitation answer for a cryptocurrency question. This suggests retrieval mismatches or context misuse, even when the final answer looks syntactically polished.
- Across all three models, the most common shared weaknesses are hallucinated legal details, incomplete answers, and answers that stay too generic instead of giving the specific tax consequence in the reference solution.

## Conclusion

In this project, the direct inference setup with `Qwen/Qwen3-0.6B` performed best under the chosen BERTScore evaluation. The fine-tuned variant showed some potential, but the lightweight CPU-friendly training setup likely limited its gains. The RAG pipeline was the weakest of the three runs, mainly because retrieval errors and context misuse had a visible negative impact on answer quality.
