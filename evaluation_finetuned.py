import json
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# CONFIG (FINETUNED)
HF_REPO_ID = "clemencetran/questcrafter-finetuned"
HF_SUBFOLDER = "fine_tuned_model"

VAL_FILE = "data/val.jsonl"
TEST_PROMPTS_FILE = "data/test_prompts.jsonl"

MAX_NEW_TOKENS = 180
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

OUT_PATH = "outputs/evaluation_finetuned.jsonl"

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, subfolder=HF_SUBFOLDER, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, subfolder=HF_SUBFOLDER).to(DEVICE)
model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# PERPLEXITY
def compute_perplexity():
    total_loss = 0.0
    total_tokens = 0

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Computing Perplexity"):
            obj = json.loads(line)
            text = obj["prompt"] + "\n" + obj["completion"]

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = inputs["input_ids"]

            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            seq_len = labels.size(1)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# GENERATE TEST SET
def generate_test_outputs():
    generations = []

    with open(TEST_PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Generating Test Set"):
            obj = json.loads(line)
            prompt = obj["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            text = tokenizer.decode(output[0], skip_special_tokens=True)

            generations.append({
                "prompt": prompt,
                "generation": text
            })

    os.makedirs("outputs", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in generations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved generations to:", OUT_PATH)
    return generations

# DISTINCT-N
def distinct_n(generations, n=1):
    total_ngrams = 0
    unique_ngrams = set()

    for g in generations:
        tokens = g["generation"].split()
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        total_ngrams += len(ngrams)
        unique_ngrams.update(ngrams)

    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

# MAIN
if __name__ == "__main__":
    print("Using device:", DEVICE)
    print("Loading fine-tuned model from:", HF_REPO_ID, "subfolder:", HF_SUBFOLDER)

    print("\n--- Perplexity ---")
    loss, ppl = compute_perplexity()
    print(f"Validation Loss: {loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")

    print("\n--- Generating Test Outputs ---")
    generations = generate_test_outputs()

    d1 = distinct_n(generations, 1)
    d2 = distinct_n(generations, 2)

    print("\n--- Diversity ---")
    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")