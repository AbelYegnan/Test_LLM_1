# test_qwen.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

print(f"transformers : {transformers.__version__}")
print(f"torch        : {torch.__version__}")

HF_ID = "Qwen/Qwen2-0.5B-Instruct"

print("\nChargement tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(HF_ID)
print("Tokenizer OK")

print("\nChargement modèle...")
model = AutoModelForCausalLM.from_pretrained(HF_ID)
model.eval()
print("Modèle OK")

print("\nTest génération...")
messages = [
    {"role": "system", "content": "Tu es un assistant utile."},
    {"role": "user",   "content": "Quelle est la valeur de HbA1c dans ce texte : HbA1c initial 8.7%, final 7.3%"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
new_tokens = out[0][inputs["input_ids"].shape[1]:]
print("Réponse :", tokenizer.decode(new_tokens, skip_special_tokens=True))