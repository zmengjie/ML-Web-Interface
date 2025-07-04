# local_llm.py

# def query_local_llm(prompt: str) -> str:
#     """
#     🦄 Dummy local-LLM stub: echoes your prompt.
#     No transformers imports here, so it will never error.
#     """
#     snippet = prompt.replace("\n", " ")[:200]
#     ellipsis = "…" if len(prompt) > 200 else ""
#     return f"🤖 [local stub] you said: “{snippet}{ellipsis}”"

# local_llm.py

from transformers import pipeline, set_seed
import streamlit as st

@st.cache_resource
def load_local_model():
    generator = pipeline("text-generation", model="distilgpt2")
    set_seed(42)
    return generator

local_model = load_local_model()

def query_local_llm(prompt: str) -> str:
    """
    Real local LLM using DistilGPT2 (transformers).
    """
    result = local_model(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]
