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


# import streamlit as st

# def query_local_llm(prompt: str) -> str:
#     """
#     Runs DistilGPT2 locally only after user submits a prompt.
#     """
#     try:
#         # Only import when needed
#         import torch
#         from transformers import pipeline, set_seed

#         # Display generation settings in sidebar
#         with st.sidebar:
#             st.markdown("### 🔧 Generation Settings (Local LLM)")
#             max_length = st.slider("Max Length", 20, 300, 100, step=10)
#             temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)

#         # Load the model only when needed
#         if "local_llm" not in st.session_state:
#             with st.spinner("🔄 Loading DistilGPT2..."):
#                 generator = pipeline("text-generation", model="distilgpt2")
#                 set_seed(42)
#                 st.session_state.local_llm = generator
#         else:
#             generator = st.session_state.local_llm

#         # Format the prompt
#         formatted_prompt = f"You are a helpful assistant.\nQ: {prompt}\nA:"
#         output = generator(
#             formatted_prompt,
#             max_length=max_length,
#             temperature=temperature,
#             num_return_sequences=1
#         )
#         return output[0]["generated_text"]

#     except ImportError:
#         return "❌ Required libraries not installed (torch or transformers)."
#     except Exception as e:
#         return f"❌ Local LLM error: {type(e).__name__}: {e}"
# local_llm.py

# import streamlit as st

# def query_local_llm(prompt: str) -> str:
#     try:
#         import torch
#         from transformers import pipeline, set_seed

#         # Sidebar sliders for controls
#         with st.sidebar:
#             st.markdown("### 🔧 Generation Settings (Local LLM)")
#             max_length = st.slider("Max Length", 20, 300, 100, step=10)
#             temperature = st.slider("Temperature", 0.1, 1.5, 0.8, step=0.1)

#         # ✅ Only load model when user queries
#         if "local_llm" not in st.session_state:
#             with st.spinner("🔄 Loading GPT2-medium..."):
#                 # generator = pipeline("text-generation", model="gpt2-medium")
#                 generator = pipeline("text-generation", model="gpt2-medium", eos_token_id=50256)
#                 set_seed(42)
#                 st.session_state.local_llm = generator
#         else:
#             generator = st.session_state.local_llm

#         formatted_prompt = f"You are a helpful assistant.\nQ: {prompt}\nA:"
#         output = generator(
#             formatted_prompt,
#             max_length=max_length,
#             temperature=temperature,
#             num_return_sequences=1
#         )
#         return output[0]["generated_text"]

#     except Exception as e:
#         return f"❌ Local LLM error: {type(e).__name__}: {e}"


# local_llm.py
import streamlit as st
import os
import requests
from ctransformers import AutoModelForCausalLM

# === Configuration ===
GGUF_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q2_K.gguf"
GGUF_PATH = "tinyllama.gguf"

# === Download model if missing ===
def download_gguf():
    if not os.path.exists(GGUF_PATH):
        print("🔽 Downloading TinyLLaMA model...")
        with requests.get(GGUF_URL, stream=True) as r:
            r.raise_for_status()
            with open(GGUF_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Download complete.")

# === Load GGUF model ===
@st.cache_resource(show_spinner="🔄 Loading local TinyLLaMA model...")
def load_local_model():
    download_gguf()
    return AutoModelForCausalLM.from_pretrained(
        GGUF_PATH,
        model_type="llama",
        gpu_layers=0,  # Set >0 if using GPU acceleration
    )

# === Initialize once ===
local_model = load_local_model()

# === Query function ===
def query_local_llm(prompt: str) -> str:
    try:
        # Use instruction-style prompt formatting
        formatted_prompt = f"### Instruction:\n{prompt.strip()}\n\n### Response:\n"
        
        full_output = local_model(formatted_prompt, max_new_tokens=200)

        # Strip unwanted response artifacts
        clean_output = full_output.replace("### Response:", "").split("###")[0].strip()

        return clean_output
    except Exception as e:
        return f"❌ Local LLM error: {e}"