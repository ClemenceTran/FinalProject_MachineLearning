import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
def load_model(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(DEVICE)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# PROMPT
def build_prompt(level: int, setting: str, tone: str, length: str, user_request: str) -> str:
    user_request = (user_request or "").strip()
    extra = f" {user_request}" if user_request else ""
    return (
        f"<LEVEL={level}> <SETTING={setting}> <TONE={tone}> <LENGTH={length}> "
        f"Write a fantasy quest story.{extra}\n"
        f"Return ONLY the quest story text. Do NOT repeat the tags or the prompt."
    )

def extract_quest_only(prompt: str, decoded_text: str) -> str:
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt):].lstrip()
    return decoded_text.strip()

# UI
st.set_page_config(page_title="QuestCrafter", page_icon="🧙")
st.title("QuestCrafter – AI Dungeon Master")

model_choice = st.selectbox("Model", ["Fine-Tuned Model", "Baseline (distilgpt2)"])
MODEL_DIR = "fine_tuned_model" if model_choice == "Fine-Tuned Model" else "distilgpt2"

level = st.selectbox("Level", list(range(1, 11)), index=4)  # default 5
setting = st.selectbox(
    "Setting",
    ["forest", "desert city", "mountain village", "ancient ruins", "seaside town"],
    index=0
)
tone = st.selectbox("Tone", ["epic", "dark", "mysterious", "humorous", "adventurous"], index=2)
length = st.selectbox("Length", ["short", "medium"], index=0)

user_request = st.text_area(
    "What quest do you want?",
    "Create a level-3 quest in a desert city with a betrayal twist."
)

max_new_tokens = st.slider("Max new tokens", 50, 300, 160, step=10)

tokenizer, model = load_model(MODEL_DIR)

if st.button("Generate Quest", type="primary"):
    prompt = build_prompt(level, setting, tone, length, user_request)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    quest_only = extract_quest_only(prompt, decoded)

    st.subheader("Generated Quest")
    st.write(quest_only)