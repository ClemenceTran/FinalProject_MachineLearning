import json
import os
import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
MODEL_NAME = "distilgpt2"
PROMPTS_PATH = "data/test_prompts.jsonl"
OUT_PATH = "outputs/baseline_generations.jsonl"

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.9
TOP_P = 0.95
REPETITION_PENALTY = 1.1

DO_SAMPLE = True
SEED = 42

# DEVICE
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = pick_device()
print("Using device:", device)

torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# LOAD MODEL 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_prompts(path: str):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts

def generate_one(prompt: str) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=TOP_P if DO_SAMPLE else None,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    gen_only = full_text[len(prompt):].lstrip() if full_text.startswith(prompt) else full_text

    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "generation_full": full_text,
        "generation_only": gen_only,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "do_sample": DO_SAMPLE,
        "seed": SEED,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

# RUN 
def main():
    if not os.path.exists(PROMPTS_PATH):
        raise FileNotFoundError(f"Missing {PROMPTS_PATH}. Put your prompts file there.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    prompts = load_prompts(PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")

    with open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for i, prompt in enumerate(prompts, start=1):
            result = generate_one(prompt)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if i % 10 == 0:
                print(f"Generated {i}/{len(prompts)}")

    print("Done! Saved baseline generations to:", OUT_PATH)

if __name__ == "__main__":
    main()