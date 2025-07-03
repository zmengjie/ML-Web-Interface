# local_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_local_model():
    model_name = "distilgpt2"  # lightweight GPT2 variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def query_local_llm(prompt):
    tokenizer, model = load_local_model()
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()
