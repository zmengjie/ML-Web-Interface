# local_llm.py
import torch
from transformers import pipeline

# this will download + cache GPT-2 (≈500 MB) the first time;
# feel free to pick any other 🤗 model you like
generator = pipeline("text-generation", model="gpt2")

def query_local_llm(prompt: str) -> str:
    """
    A very simple local LLM: feeds `prompt` into GPT-2
    and returns the generated text (up to 200 tokens).
    """
    out = generator(
        prompt,
        max_length=len(prompt.split()) + 100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    # strip away the prompt itself
    return out[0]["generated_text"][len(prompt) :].strip()
