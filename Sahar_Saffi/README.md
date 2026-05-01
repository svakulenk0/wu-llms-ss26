# Project Report – Austrian Tax Law

**Author:** Sahar Saffi
**Course:** LLMs – WU Vienna

# Models

# Model 1 – Inference Model
I used the  model via the Google Gemini API. No fine-tuning, just a system prompt telling the model to answer Austrian tax questions in German and reference the relevant laws. Temperature was set to 0.2 and max tokens to 400. The model worked well but sometimes returned 503 errors when the API was overloaded, so it took me much time.

# Model 2 – flan-t5-base (Fine-tuned)
I fine-tuned flan-t5-base (247M parameters) on 3 manually written Q&A pairs about Austrian tax law. I only had 3 examples because I was running on CPU and it was already slow. The loss went from around 4.4 down to 3.7 which shows it learned something, but 3 examples is obviously not a lot. If I had more time on training it with more examples, it would have been better.

# Model 3 – TF-IDF + flan-t5-base (RAG)
I built a simple retrieval system using TF-IDF over 10 manually written chunks from Austrian tax laws (EStG, KStG, UStG, BAO, GrEStG). For each question the system finds the most similar chunk and returns it as the answer. It's not perfect but it always returns something factual from the actual laws.

# Evaluation
I used Model 1 as the reference and compared the other two against it with ROUGE and BLEU.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |

|---|---|---|---|---|

| Model 1 – Gemma 3 27B | Reference | Reference | Reference | Reference |
| Model 2 – flan-t5-base fine-tuned | 0.3794 | 0.1741 | 0.2596 | 0.1116 |
| Model 3 – RAG | 0.0548 | 0.0048 | 0.0438 | 0.0002 |

Model 1 is clearly the best. Model 2 is okay given it was only trained on 3 examples. Model 3 scores really low because it returns raw legal text.

# Error Analysis
Model 1 – a few empty answers due to API errors.
Model 2 – answers are too short and generic, the model couldn't really generalize from just 3 training examples.
Model 3 – the retrieval often picks the wrong law chunk, and returning raw text as an answer is not great for this kind of task.

All three models had trouble with questions that involve multiple laws at the same time. So i had to do try it many times and unfortuntely did the whole task twice. 