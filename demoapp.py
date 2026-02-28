import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
HF_REPO_ID = "clemencetran/questcrafter-finetuned"
HF_SUBFOLDER = "fine_tuned_model" 

SETTINGS = ["forest", "desert city", "mountain village", "ancient ruins", "seaside town"]
TONES = ["epic", "dark", "mysterious", "humorous", "adventurous"]
LEVELS = list(range(1, 11))
LENGTHS = ["short", "medium"]

# DEVICE
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()

# MODEL LOADER
@st.cache_resource
def load_model(model_choice: str):
    """
    Loads either:
      - Fine-Tuned from your Hugging Face repo subfolder
      - Baseline from 'distilgpt2'
    """
    if model_choice == "Fine-Tuned":
        model_id = HF_REPO_ID
        tok_kwargs = {"subfolder": HF_SUBFOLDER}
        mdl_kwargs = {"subfolder": HF_SUBFOLDER}
    else:
        model_id = "distilgpt2"
        tok_kwargs = {}
        mdl_kwargs = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **tok_kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **tok_kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_id, **mdl_kwargs).to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# PROMPT HELPERS
def build_prompt(level: int, setting: str, tone: str, length: str, user_request: str) -> str:
    user_request = (user_request or "").strip()
    extra = f" {user_request}" if user_request else ""
    return (
        f"<LEVEL={level}> <SETTING={setting}> <TONE={tone}> <LENGTH={length}> "
        f"Write a fantasy quest story.{extra}\n"
        f"Return ONLY the quest story text. Do NOT repeat the tags or the prompt."
    )

def extract_quest_only(prompt: str, decoded_text: str) -> str:
    decoded_text = decoded_text.strip()
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt):].lstrip()
    return decoded_text

# UI
st.set_page_config(page_title="QuestCrafter", page_icon="🧙")
st.title("QuestCrafter – AI Dungeon Master")
st.caption(f"Device: {DEVICE}")

model_choice = st.selectbox("Model", ["Fine-Tuned", "Baseline"])

level = st.selectbox("Level", LEVELS, index=4)
setting = st.selectbox("Setting", SETTINGS, index=0)
tone = st.selectbox("Tone", TONES, index=2)
length = st.selectbox("Length", LENGTHS, index=0)

user_request = st.text_area(
    "Additional quest instructions (optional)",
    "Create a quest with a betrayal twist."
)

max_new_tokens = st.slider("Max new tokens", 50, 300, 160, step=10)

tokenizer, model = load_model(model_choice)

if st.button("Generate Quest", type="primary"):
    prompt = build_prompt(level, setting, tone, length, user_request)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    quest_only = extract_quest_only(prompt, decoded)

    st.subheader("Generated Quest")
    st.write(quest_only)