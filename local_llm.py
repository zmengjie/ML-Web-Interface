# local_llm.py

def query_local_llm(prompt: str) -> str:
    """
    🦄 Dummy local-LLM stub: echoes your prompt.
    No transformers imports here, so it will never error.
    """
    snippet = prompt.replace("\n", " ")[:200]
    ellipsis = "…" if len(prompt) > 200 else ""
    return f"🤖 [local stub] you said: “{snippet}{ellipsis}”"
