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

# local_llm.py

def query_local_llm(prompt: str) -> str:
    # Echo‐stub: just returns the first 200 characters of the prompt
    snippet = prompt.replace("\n", " ")[:200]
    return f"🤖 [local stub] you said: “{snippet}{'…' if len(prompt) > 200 else ''}”"

