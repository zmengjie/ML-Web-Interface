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

def query_local_llm(prompt: str) -> str:
    """
    Runs local DistilGPT2 for text generation.
    Loads model only when first called.
    """
    try:
        from transformers import pipeline, set_seed
        generator = st.session_state.get("local_llm")
        if generator is None:
            with st.spinner("🔄 Loading local model (DistilGPT2)..."):
                generator = pipeline("text-generation", model="distilgpt2")
                set_seed(42)
                st.session_state.local_llm = generator

        result = generator(prompt, max_length=100, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        return f"❌ Local LLM error: {e}"