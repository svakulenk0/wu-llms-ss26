# Project Summary

This project uses **LLM-as-a-Judge** to evaluate AI-generated answers in Austrian tax law.

## Setup

I initially planned to use Mixtral via Hugging Face, but due to connectivity issues, I switched to **GPT-4o-mini** as an API-based judge. I designed a strict `JUDGE_PROMPT` to simulate an expert tax auditor.

**Key choices:**

* Focus only on Austrian tax law (EStG, UStG, BAO)
* Penalize hallucinations and incorrect legal claims
* Cap wrong answers at a score of 2
* Reward precision over verbosity
* Use a 1-4 scale to force clear judgments

## Key Insights

* **Fluency ≠ correctness:** The inference model sounded best but made legal errors
* **Style ≠ understanding:** The fine-tuned model mimicked expert language but hallucinated
* **Safety matters:** The RAG model refused uncertain answers, making it more reliable

## Conclusion

Standard metrics like BERTScore are not enough for legal AI.
While fine-tuning improves style, **RAG showed the most reliable behavior by prioritizing correctness over confidence**.

