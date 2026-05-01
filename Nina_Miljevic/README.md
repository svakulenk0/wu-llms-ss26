# Nina Miljevic – LLM Assignment Submission

This project evaluates three different approaches for answering Austrian tax law questions:

1. Inference-only (API-based)
2. Fine-tuned model
3. Retrieval-Augmented Generation (RAG)



## 1. Inference

- Model: LLaMA 3.1 (via Groq API)
- Approximate size: ~8B parameters
- Pretraining data: large-scale multilingual web data, books, and code (general-purpose LLM)

### Hyperparameters
- Temperature: 0 (deterministic output, no randomness)
- Max tokens: 120
- Sampling: disabled (greedy decoding - model looks at all possible next tokens and picks the most probable one)

### Approach
The model directly answers the question without additional training (fine-tuning) or external knowledge retrieval.

### Files
- Code: `code/code_inference.py`
- Output: `results/results_inference_groq_clean.csv`

---

## 2. Fine-tuned Model

- Base model: Qwen2.5-1.5B-Instruct
- Approximate size: 1.5B parameters
- Pretraining data: multilingual web and instruction-tuning datasets

### Fine-tuning Data
- Custom dataset (~40 question–answer pairs, created by myself)
- Domain: Austrian tax law
- Important: The provided test dataset was NOT used for training/fine-tuning

### Fine-tuning Method
- Supervised fine-tuning using LoRA (parameter-efficient tuning - instead of changing the whole model, it makes low-rank updates to the model)
- Framework: Hugging Face Transformers + TRL
- Environment: Google Colab (T4 GPU)

### Hyperparameters
- Epochs = how many times the model sees the training dataset: 3
- Batch size = how many examples are being processed at the same time: 1 (with gradient accumulation)
- Learning rate = how big each update step is during training: 2e-4
- LoRA rank (r) = how much the model can adapt: 8
- LoRA alpha = scaling factor for LoRA updates, controls how the strong the adjusments are: 16
- Dropout = regularization technique to reduce overfitting and improve generalization by "turning off" some of the neurons, in this case 5% of neurons: 0.05

### Observations
Due to the small size of the training dataset, the model shows limited generalization ability.

### Files
- Code: `code/code_fine-tune_google_colab.py`
- Output: `results/results_finetuned_qwen.csv`

---

## 3. Retrieval-Augmented Generation (RAG)

### Architecture
The RAG system combines:
- Retrieval of relevant legal passages
- Generation using a large language model

### Retrieval Model
- Embedding model: SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- Vector database: FAISS (L2 similarity search)

### Document Collection
The following Austrian legal sources were used:
- Einkommensteuergesetz (EStG)
- Körperschaftsteuergesetz (KStG)
- Umsatzsteuergesetz (UStG)
- Bundesabgabenordnung (BAO)
- Unternehmensgesetzbuch (UGB)
- Grunderwerbsteuergesetz (GrEStG)

### Preprocessing
- PDFs were converted to plain text
- Text was split into chunks (~500 characters each)
- Very short or empty chunks were removed

### Retrieval Setup
- Top-k retrieval: 3 passages per query
- Retrieved passages are concatenated and used as context

### Generation Model
- Model: LLaMA 3.1 via Groq API
- Temperature: 0
- Max tokens: 150

### Files
- Code: `code/code_rag.py`
- Output: `results/results_rag_groq.csv`

---

## Evaluation

### Metrics

In order to evaluate how each of the models peformed at retrieving the right answer, automated metrics were computed.

The following automated metrics were used:
- BLEU (n-gram overlap)
- ROUGE-L (longest common subsequence)
- BERTScore (semantic similarity)

### Results

| Model        | BLEU  | ROUGE-L | BERTScore |
|-------------|------|---------|-----------|
| Inference   | 0.028 | 0.171 | 0.710 |
| Fine-tuned  | 0.016 | 0.130 | 0.675 |
| RAG         | 0.037 | 0.186 | 0.715 |

### Interpretation

Based on the results it can concluded that the RAG-based model achieved the best performance across all evaluation metrics. This indicates that providing the model with relevant legal context significantly improves answer quality. 

The inference-only model performed moderately well, benefiting from a strong pretrained model but lacking domain-specific grounding.

The fine-tuned model performed worst, likely due to the small size of the training dataset, which limited its ability to generalize.

### Metric considerations

BLEU scores are relatively low for all models. This is expected because BLEU relies on exact word overlap and is not well-suited for legal question-answering tasks. It looks at the number of n-grams in the generated sentence and checks how many of them are in the right answer. It penalizes reordered and missing n-grams. Therefore it will give a bad score, even if the sentence is true but is paraphrased.

BERTScore is more informative, as it captures semantic similarity between generated and reference answers using embeddings. 

---

## Error Analysis

To better understand model performance, a qualitative error analysis was conducted on low-scoring examples (based on BERTScore).

### Common Error Types that I Detected

#### 1. Lack of Knowledge / Outdated Information
Some models refused to answer and instead stated that the information was unavailable or outdated.

#### 2. Semantic Misinterpretation
Models sometimes confused legal concepts and produced unrelated answers. 

#### 3. Generic Responses
Some outputs contained generic advice (e.g., consulting a tax advisor), which is not useful for this task.

### Comparison Across Models

These error types were most prominent in the inference-based model.

The fine-tuned model reduced some of these issues but still struggled with accuracy due to limited training data.

The RAG-based model showed the fewest errors of this type, as it was able to rely on retrieved legal context, resulting in more precise and relevant answers.

### Conclusion from Error Analysis

The error analysis confirms that:
- Pretrained models without grounding tend to hallucinate or avoid answering.
- Small-scale fine-tuning is not sufficient to fully capture domain-specific knowledge.
- Retrieval-Augmented Generation significantly improves answer quality by grounding responses in relevant legal texts.

---

## Reproducibility

All code required to reproduce the results is included in the `code` folder.

The test dataset was provided by the course and was not used for training!

Due to file size and copyright considerations, the original legal PDFs that I used for the Retrieval-Augmented Generation Model are not included in this repository. The documents were obtained from Jusline (https://www.jusline.at/gesetzesbibliothek). To ensure computational efficiency, only the relevant legal passages were extracted and processed, rather than the full texts of the laws.