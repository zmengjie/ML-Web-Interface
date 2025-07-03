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

def query_local_llm(prompt: str) -> str:
    out = generator(
        prompt,
        max_length=len(prompt.split()) + 50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    # remove the prompt prefix
    generated = out[0]["generated_text"]
    return generated[len(prompt):].strip()
