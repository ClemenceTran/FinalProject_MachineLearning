import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# CONFIG
MODEL_NAME = "distilgpt2"
TRAIN_FILE = "data/train.jsonl"
OUTPUT_DIR = "fine_tuned_model"

MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5

# DEVICE
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = pick_device()
print("Using device:", device)

# LOAD DATA
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt = obj["prompt"]
            completion = obj["completion"]

            text = prompt + "\n" + completion
            data.append({"text": text})

    return Dataset.from_list(data)

dataset = load_jsonl(TRAIN_FILE)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# MODEL
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,

    logging_steps=50,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# TRAIN
trainer.train()

# SAVE
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved to:", OUTPUT_DIR)