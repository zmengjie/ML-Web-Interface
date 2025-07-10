
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


# import streamlit as st
# import os
# import requests
# from ctransformers import AutoModelForCausalLM

# try:
#     from rag_retriever import retrieve_relevant_chunks
# except ImportError as e:
#     print(f"⚠️ Failed to import RAG module: {e}")
#     def retrieve_relevant_chunks(query, top_k=3): return []


# # === Configuration ===
# GGUF_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
# GGUF_PATH = "tinyllama-q4.gguf"

# # === Global setting ===
# MODEL_FORMAT = "tinyllama"

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

# @st.cache_resource(show_spinner="🔄 Loading Orca Mini 3B...")
# def load_local_model():
#     download_gguf()
#     return AutoModelForCausalLM.from_pretrained(
#         GGUF_PATH,
#         model_type="llama",  # Orca Mini is LLaMA-based
#         gpu_layers=0
#     )

# # === Initialize model ===
# local_model = load_local_model()

# # === Query function with RAG ===
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


import os
import streamlit as st
from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM

# === Optional: RAG retriever ===
try:
    from rag_retriever import retrieve_relevant_chunks
except ImportError:
    def retrieve_relevant_chunks(query, top_k=3): return []

# === GGUF Configuration ===
MISTRAL_REPO = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
MISTRAL_FILE = "mistral-7b-instruct-v0.1.Q4_0.gguf"
MISTRAL_PATH = os.path.join(".", MISTRAL_FILE)

TINYLLAMA_REPO = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF"
TINYLLAMA_FILE = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
TINYLLAMA_PATH = os.path.join(".", TINYLLAMA_FILE)

# === Download if missing ===
def download_model(repo_id, filename, save_path):
    if not os.path.exists(save_path):
        print(f"🔽 Downloading {filename}...")
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        os.rename(model_file, save_path)
        print("✅ Download complete.")

# === Prompt formatting ===
def format_prompt(prompt: str) -> str:
    return f"[INST] {prompt.strip()} [/INST]"

# === Clean and truncate output ===
def clean_output(raw: str) -> str:
    stop_tokens = ["[/INST]", "###", "<|", "Wings of Glass", "I've been working on"]
    for token in stop_tokens:
        if token in raw:
            raw = raw.split(token)[0]
    return raw.strip()

# === Check if answer matches context ===
def is_answer_contextual(answer: str, context: str) -> bool:
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    return len(answer_words & context_words) >= 3

# === Load models ===
@st.cache_resource(show_spinner="🔄 Loading Mistral 7B...")
def load_mistral():
    download_model(MISTRAL_REPO, MISTRAL_FILE, MISTRAL_PATH)
    return AutoModelForCausalLM.from_pretrained(
        MISTRAL_PATH,
        model_type="mistral",
        gpu_layers=0
    )

@st.cache_resource(show_spinner="🔄 Loading TinyLLaMA...")
def load_tinyllama():
    download_model(TINYLLAMA_REPO, TINYLLAMA_FILE, TINYLLAMA_PATH)
    return AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_PATH,
        model_type="llama",
        gpu_layers=0
    )

# === Global instances ===
mistral_model = load_mistral()
tinyllama_model = load_tinyllama()

# === Main query function ===
def query_local_llm(prompt: str) -> str:
    try:
        context_chunks = retrieve_relevant_chunks(prompt, top_k=3)
        context_str = "\n\n".join(context_chunks)

        # Prompt for Mistral
        base_prompt = (
            "You are a helpful assistant that only answers based on the given context. "
            "If the answer is not in the context, reply with: 'I don't know based on the provided context.'\n\n"
            f"### Context:\n{context_str}\n\n"
            f"### Question:\n{prompt.strip()}\n\n"
            f"### Answer:\n"
        )
        mistral_input = format_prompt(base_prompt)

        MAX_TOKENS = 2048
        RESERVED_TOKENS = 400
        max_prompt_words = int((MAX_TOKENS - RESERVED_TOKENS) / 1.3)
        words = mistral_input.strip().split()
        if len(words) > max_prompt_words:
            mistral_input = " ".join(words[-max_prompt_words:])
            st.warning(f"⚠️ Mistral prompt trimmed to last {max_prompt_words} words.")

        st.markdown("📚 **Retrieved Context:**")
        st.code(context_str, language='text')
        st.code(mistral_input, language='text')

        # Mistral inference
        mistral_output = mistral_model(mistral_input, max_new_tokens=RESERVED_TOKENS)
        cleaned = clean_output(mistral_output)

        if cleaned and is_answer_contextual(cleaned, context_str):
            return cleaned

        # ⛑ Fallback: TinyLLaMA (if Mistral failed or context empty)
        fallback_prompt = f"Answer this question briefly: {prompt.strip()}"
        tiny_output = tinyllama_model(format_prompt(fallback_prompt), max_new_tokens=300)
        return clean_output(tiny_output)

    except Exception as e:
        return f"❌ Local LLM error: {e}"
