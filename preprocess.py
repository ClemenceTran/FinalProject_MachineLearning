import random
import json
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# CONFIG
MAX_SAMPLES = 100000
OUTPUT_DIR = "data"

MIN_WORDS = 30
MAX_WORDS = 250

SEED = 42
random.seed(SEED)

# TOKENS
SETTINGS = ["forest", "desert city", "mountain village", "ancient ruins", "seaside town"]
TONES = ["epic", "dark", "mysterious", "humorous", "adventurous"]
LEVELS = list(range(1, 11))
LENGTHS = ["short", "medium"]

def build_prompt():
    level = random.choice(LEVELS)
    setting = random.choice(SETTINGS)
    tone = random.choice(TONES)
    length = random.choice(LENGTHS)

    return (
        f"<LEVEL={level}> <SETTING={setting}> "
        f"<TONE={tone}> <LENGTH={length}> "
        f"Write a fantasy quest story."
    )

# LOAD DATASET
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

print(f"Shuffling and selecting {MAX_SAMPLES} samples...")
dataset = dataset.shuffle(seed=SEED).select(range(MAX_SAMPLES))

# CLEAN + FORMAT
print("Cleaning + formatting...")
data = []
for ex in dataset:
    story = ex["text"].strip()
    wc = len(story.split())

    if wc < MIN_WORDS or wc > MAX_WORDS:
        continue

    prompt = build_prompt()

    data.append({
        "prompt": prompt,
        "completion": story
    })

print("Clean samples:", len(data))

# SPLIT DATA -> TRAIN, VAL, TEST (80/10/10)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

# SAVE
def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_jsonl(train_data, f"{OUTPUT_DIR}/train.jsonl")
save_jsonl(val_data, f"{OUTPUT_DIR}/val.jsonl")
save_jsonl(test_data, f"{OUTPUT_DIR}/test.jsonl")

print("Done!")
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))
print(f"Saved to: {OUTPUT_DIR}/")