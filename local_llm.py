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



import streamlit as st

# ✅ Explicitly import both transformers and torch
try:
    import torch
    from transformers import pipeline, set_seed
except ImportError as e:
    st.error("❌ Required libraries not available. Please install torch and transformers.")
    raise e

def query_local_llm(prompt: str) -> str:
    """
    Runs local DistilGPT2 for text generation using transformers + torch.
    Loads once and stores in session_state.
    """
    try:
        if "local_llm" not in st.session_state:
            with st.spinner("🔄 Loading local model (DistilGPT2)..."):
                generator = pipeline("text-generation", model="distilgpt2")
                set_seed(42)
                st.session_state.local_llm = generator
        else:
            generator = st.session_state.local_llm

        output = generator(prompt, max_length=100, num_return_sequences=1)
        return output[0]["generated_text"]

    except Exception as e:
        return f"❌ Local LLM error: {type(e).__name__}: {e}"
