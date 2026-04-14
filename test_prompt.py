from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

questions = [
    "Was ist die steuerliche Bemessungsgrundlage für die Körperschaftsteuer?",
    "Welche Körperschaften sind verpflichtet, sämtliche Gebühren der Gebühren aus Gewerbebetrieb zu zahlen?"
]

prompts = [
    "Beantworte die folgende Frage zum österreichischen Steuerrecht auf Deutsch: {q}",
    "Frage: {q}\nAntwort:",
    "Answer the following question in German.\n\nQuestion: {q}\n\nAnswer:"
]

for p_tmpl in prompts:
    print(f"--- Prompt: {p_tmpl.repr() if hasattr(p_tmpl, 'repr') else repr(p_tmpl)} ---")
    for q in questions:
        p = p_tmpl.format(q=q)
        inputs = tokenizer(p, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50)
        print("  Q:", q)
        print("  A:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        print()

