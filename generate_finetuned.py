import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Correct repo + subfolder
MODEL_ID = "clemencetran/questcrafter-finetuned"
SUBFOLDER = "fine_tuned_model"

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = pick_device()
print("Using device:", device)

# IMPORTANT: include subfolder
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    subfolder=SUBFOLDER,
    use_fast=False   # safer
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    subfolder=SUBFOLDER
).to(device)

model.eval()

prompt = "<LEVEL=5> <SETTING=forest> <TONE=dark> <LENGTH=short> Write a fantasy quest story."

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result)