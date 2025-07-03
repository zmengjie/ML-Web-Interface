# local_llm.py
import torch
from transformers import pipeline

# instantiate the generator once at import time
generator = pipeline(
    "text-generation",
    model="gpt2",
    # explicitly tell it which torch dtype to use:
    torch_dtype=torch.float32,
)

# local_llm.py

def query_local_llm(prompt: str) -> str:
    """
    🦄 Dummy local‐LLM stub: echoes back up to 200 chars of the prompt.
    Replace this with real inference later once your dependencies are sorted.
    """
    snippet = prompt.replace("\n", " ")[:200]
    ellipsis = "…" if len(prompt) > 200 else ""
    return f"🤖 [local stub] you said: “{snippet}{ellipsis}”"