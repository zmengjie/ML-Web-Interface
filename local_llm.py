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

# local_llm.py

import streamlit as st

# Fully pre-import torch and transformers to avoid lazy errors
try:
    import torch
    from transformers import pipeline, set_seed
except ImportError as e:
    st.error("❌ Required library missing. Please ensure 'torch' and 'transformers' are installed.")
    raise

def get_generator():
    """
    Load the DistilGPT2 model safely and only once.
    """
    if "local_llm" not in st.session_state:
        try:
            with st.spinner("🔄 Loading local LLM (DistilGPT2)..."):
                generator = pipeline("text-generation", model="distilgpt2")
                set_seed(42)
                st.session_state.local_llm = generator
        except Exception as e:
            st.session_state.local_llm = None
            st.error(f"❌ Failed to load local LLM: {e}")
            raise
    return st.session_state.local_llm

def query_local_llm(prompt: str) -> str:
    """
    Generate text using local DistilGPT2 model.
    """
    try:
        generator = get_generator()
        if generator is None:
            return "⚠️ Local LLM is not available."
        output = generator(prompt, max_length=100, num_return_sequences=1)
        return output[0]["generated_text"]
    except Exception as e:
        return f"❌ Local LLM error: {type(e).__name__}: {e}"
