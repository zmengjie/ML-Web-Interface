
# # local_llm.py
# import streamlit as st
# import os
# import requests
# from ctransformers import AutoModelForCausalLM

# # === Configuration ===
# GGUF_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
# GGUF_PATH = "tinyllama-q4.gguf"


# # === Global setting ===
# MODEL_FORMAT = "tinyllama"  # Change to "mistral" or "llama3" when upgrading



# # === Prompt formatter ===
# def format_prompt(prompt: str) -> str:
#     prompt = prompt.strip()
#     if MODEL_FORMAT == "tinyllama":
#         return f"### Instruction:\n{prompt}\n\n### Response:\n"
#     elif MODEL_FORMAT == "mistral":
#         return f"[INST] {prompt} [/INST]"
#     elif MODEL_FORMAT == "llama3":
#         return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
#     else:
#         return prompt  # fallback

# # === Clean model output ===
# def clean_output(raw: str) -> str:
#     if MODEL_FORMAT == "llama3":
#         raw = raw.replace("<|im_start|>", "").replace("<|im_end|>", "")
#     elif MODEL_FORMAT == "tinyllama":
#         raw = raw.replace("### Response:", "")
#     return raw.split("###")[0].strip()

# # === Download model if missing ===
# def download_gguf():
#     if not os.path.exists(GGUF_PATH):
#         print("🔽 Downloading TinyLLaMA model...")
#         with requests.get(GGUF_URL, stream=True) as r:
#             r.raise_for_status()
#             with open(GGUF_PATH, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print("✅ Download complete.")

# # === Load GGUF model ===
# @st.cache_resource(show_spinner="🔄 Loading TinyLLaMA Q4_K_M...")
# def load_local_model():
#     download_gguf()
#     return AutoModelForCausalLM.from_pretrained(
#         GGUF_PATH,
#         model_type="llama",
#         gpu_layers=0,  # Set >0 if using GPU acceleration
#     )



# # === Initialize once ===
# local_model = load_local_model()

# # === Query function ===
# def query_local_llm(prompt: str) -> str:
#     try:
#         # === Format prompt ===
#         formatted_prompt = format_prompt(prompt)

#         # === Token trimming ===
#         MAX_TOKENS = 2048
#         RESERVED_TOKENS = 400
#         max_prompt_words = int((MAX_TOKENS - RESERVED_TOKENS) / 1.3)

#         words = formatted_prompt.strip().split()
#         if len(words) > max_prompt_words:
#             formatted_prompt = " ".join(words[-max_prompt_words:])
#             st.warning(f"⚠️ Prompt was too long and has been trimmed to the last {max_prompt_words} words.")

#         # === Show prompt and response in Streamlit ===
#         st.code(formatted_prompt, language='text')
#         full_output = local_model(formatted_prompt, max_new_tokens=RESERVED_TOKENS)
#         st.text("🧠 Raw output:\n" + full_output)

#         # === Clean output ===
#         clean_output_text = clean_output(full_output)
#         return clean_output_text  # ✅ FIXED LINE

#     except Exception as e:
#         return f"❌ Local LLM error: {e}"


# local_llm.py
# import streamlit as st
# import os
# import requests
# from ctransformers import AutoModelForCausalLM

# # === Configuration ===
# GGUF_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# GGUF_PATH = "mistral-7b-instruct-q4.gguf"

# # === Global setting ===
# MODEL_FORMAT = "mistral"  # <- Set to "mistral"


# # === Prompt formatter ===
# def format_prompt(prompt: str) -> str:
#     prompt = prompt.strip()
#     if MODEL_FORMAT == "tinyllama":
#         return f"### Instruction:\n{prompt}\n\n### Response:\n"
#     elif MODEL_FORMAT == "mistral":
#         return f"[INST] {prompt} [/INST]"
#     elif MODEL_FORMAT == "llama3":
#         return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
#     else:
#         return prompt  # fallback

# # === Clean model output ===
# def clean_output(raw: str) -> str:
#     if MODEL_FORMAT == "llama3":
#         raw = raw.replace("<|im_start|>", "").replace("<|im_end|>", "")
#     elif MODEL_FORMAT == "mistral":
#         raw = raw.split("[/INST]")[-1].strip()
#     return raw.split("###")[0].strip()

# # === Download model if missing ===
# def download_gguf():
#     if not os.path.exists(GGUF_PATH):
#         print("🔽 Downloading Mistral-7B-Instruct model...")
#         with requests.get(GGUF_URL, stream=True) as r:
#             r.raise_for_status()
#             with open(GGUF_PATH, 'wb') as f:
#                 for chunk in r.iter_content(chunk_size=8192):
#                     f.write(chunk)
#         print("✅ Download complete.")

# # === Load GGUF model ===

# @st.cache_resource(show_spinner="🔄 Loading Mistral-7B-Instruct Q4_K_M...")
# def load_local_model():
#     download_gguf()
#     return AutoModelForCausalLM.from_pretrained(
#         GGUF_PATH,
#         model_type="mistral",  # <---- Important!
#         gpu_layers=0,         # adjust based on your GPU
#     )


# # === Initialize once ===
# local_model = load_local_model()

# # === Query function ===
# def query_local_llm(prompt: str) -> str:
#     try:
#         # === Format prompt ===
#         formatted_prompt = format_prompt(prompt)

#         # === Token trimming ===
#         MAX_TOKENS = 2048
#         RESERVED_TOKENS = 400
#         max_prompt_words = int((MAX_TOKENS - RESERVED_TOKENS) / 1.3)

#         words = formatted_prompt.strip().split()
#         if len(words) > max_prompt_words:
#             formatted_prompt = " ".join(words[-max_prompt_words:])
#             st.warning(f"⚠️ Prompt was too long and has been trimmed to the last {max_prompt_words} words.")

#         # === Show prompt and response in Streamlit ===
#         st.code(formatted_prompt, language='text')
#         full_output = local_model(formatted_prompt, max_new_tokens=RESERVED_TOKENS)
#         st.text("🧠 Raw output:\n" + full_output)

#         # === Clean output ===
#         clean_output_text = clean_output(full_output)
#         return clean_output_text  # ✅ FIXED LINE

#     except Exception as e:
#         return f"❌ Local LLM error: {e}"


import streamlit as st
import os
import requests
from ctransformers import AutoModelForCausalLM

try:
    from rag_retriever import retrieve_relevant_chunks
except ImportError as e:
    print(f"⚠️ Failed to import RAG module: {e}")
    def retrieve_relevant_chunks(query, top_k=3): return []


# === Configuration ===
# GGUF_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
# GGUF_PATH = "tinyllama-q4.gguf"

# # === Global setting ===
# MODEL_FORMAT = "tinyllama"


# === Use Phi-2 instead ===
GGUF_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
GGUF_PATH = "phi2-q4.gguf"
MODEL_FORMAT = "mistral"  # or "llama3" if needed


# === Prompt formatter ===
def format_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if MODEL_FORMAT == "tinyllama":
        return f"### Instruction:\n{prompt}\n\n### Response:\n"
    elif MODEL_FORMAT == "mistral":
        return f"[INST] {prompt} [/INST]"
    elif MODEL_FORMAT == "llama3":
        return f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        return prompt  # fallback

# === Clean model output ===
def clean_output(raw: str) -> str:
    if MODEL_FORMAT == "llama3":
        raw = raw.replace("<|im_start|>", "").replace("<|im_end|>", "")
    elif MODEL_FORMAT == "tinyllama":
        raw = raw.replace("### Response:", "")
    return raw.split("###")[0].strip()

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
@st.cache_resource(show_spinner="🔄 Loading Phi-2 Q4_K_M...")
def load_local_model():
    download_gguf()
    return AutoModelForCausalLM.from_pretrained(
        GGUF_PATH,
        model_type="gpt2",
        gpu_layers=0,  # Set >0 if GPU memory available
    )

# === Initialize model ===
local_model = load_local_model()

# === Query function with RAG ===
# def query_local_llm(prompt: str) -> str:
#     try:
#         # === Retrieve relevant chunks from local vector DB ===
#         context_chunks = retrieve_relevant_chunks(prompt, top_k=3)
#         context_str = "\n\n".join(context_chunks)

#         # === Build RAG-enhanced prompt ===
#         full_prompt = (
#             f"You are a helpful assistant. Use the context below to answer the question.\n\n"
#             f"### Context:\n{context_str}\n\n"
#             f"### Question:\n{prompt}\n\n"
#             f"### Answer:\n"
#         )

#         # === Format prompt ===
#         formatted_prompt = format_prompt(full_prompt)

#         # === Token trimming ===
#         MAX_TOKENS = 2048
#         RESERVED_TOKENS = 400
#         max_prompt_words = int((MAX_TOKENS - RESERVED_TOKENS) / 1.3)

#         words = formatted_prompt.strip().split()
#         if len(words) > max_prompt_words:
#             formatted_prompt = " ".join(words[-max_prompt_words:])
#             st.warning(f"⚠️ Prompt was too long and trimmed to last {max_prompt_words} words.")

#         # === Optional: show context ===
#         st.markdown("📚 **Retrieved Context:**")
#         st.code(context_str, language='text')

#         # === Show prompt and response ===
#         st.code(formatted_prompt, language='text')
#         full_output = local_model(formatted_prompt, max_new_tokens=RESERVED_TOKENS)
#         st.text("🧠 Raw output:\n" + full_output)

#         # === Clean output ===
#         clean_output_text = clean_output(full_output)
#         return clean_output_text

#     except Exception as e:
#         return f"❌ Local LLM error: {e}"

def clean_output(raw: str) -> str:
    """Clean LLM output and truncate after irrelevant or hallucinated segments."""
    if MODEL_FORMAT == "llama3":
        raw = raw.replace("<|im_start|>", "").replace("<|im_end|>", "")
    elif MODEL_FORMAT == "tinyllama":
        raw = raw.replace("### Response:", "")
    elif MODEL_FORMAT == "mistral":
        raw = raw.replace("[/INST]", "")

    # Truncate at known stop phrases or nonsense drift
    stop_tokens = ["###", "Instruction", "Context", "Wings of Glass", "I've been working on"]
    for token in stop_tokens:
        if token in raw:
            raw = raw.split(token)[0]

    return raw.strip()


def is_answer_contextual(answer: str, context: str) -> bool:
    """Simple keyword check to see if the answer uses any of the context."""
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = answer_words & context_words
    return len(overlap) >= 3  # threshold can be adjusted


def query_local_llm(prompt: str) -> str:
    try:
        # === RAG retrieval ===
        context_chunks = retrieve_relevant_chunks(prompt, top_k=3)
        context_str = "\n\n".join(context_chunks)

        # === Construct a strict instruction-following prompt ===
        base_prompt = (
            "You are a helpful assistant that only answers based on the given context. "
            "If the answer is not in the context, reply exactly with: 'I don't know based on the provided context.'\n\n"
            f"### Context:\n{context_str}\n\n"
            f"### Question:\n{prompt}\n\n"
            f"### Answer:\n"
        )
        formatted_prompt = format_prompt(base_prompt)

        # === Trim if too long ===
        MAX_TOKENS = 2048
        RESERVED_TOKENS = 400
        max_prompt_words = int((MAX_TOKENS - RESERVED_TOKENS) / 1.3)
        words = formatted_prompt.strip().split()
        if len(words) > max_prompt_words:
            formatted_prompt = " ".join(words[-max_prompt_words:])
            st.warning(f"⚠️ Prompt trimmed to {max_prompt_words} words.")

        # === Display context and prompt ===
        st.markdown("📚 **Retrieved Context:**")
        st.code(context_str, language='text')
        st.code(formatted_prompt, language='text')

        # === Generate output ===
        full_output = local_model(formatted_prompt, max_new_tokens=RESERVED_TOKENS)
        st.text("🧠 Raw output:\n" + full_output)

        # === Clean output ===
        cleaned = clean_output(full_output)

        # === Reject hallucinations ===
        if not is_answer_contextual(cleaned, context_str):
            return "⚠️ I don't know based on the provided context."

        return cleaned or "⚠️ No valid response returned."

    except Exception as e:
        return f"❌ Local LLM error: {e}"
