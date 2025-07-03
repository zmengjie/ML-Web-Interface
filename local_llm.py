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
    try:
        from transformers import pipeline
        # Only load the model when this function is called:
        generator = pipeline(
            "text-generation",
            model="gpt2",
            # force CPU, no dtype inference
            device=-1,  
        )
        out = generator(prompt, max_length=100, do_sample=False)
        return out[0]["generated_text"]
    except Exception as e:
        return f"[Local LLM failed to load: {e}]"
