import os
import faiss
import pickle
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

# === Paths ===
FAISS_INDEX_PATH = "rag_index.faiss"
DOC_STORE_PATH = "rag_docs.pkl"

# === Encoder ===
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# === Build or load index ===
def build_index(text_chunks: List[str]):
    embeddings = encoder.encode(text_chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
    faiss.write_index(index, FAISS_INDEX_PATH)


def retrieve_chunks(query: str, k: int = 3) -> List[str]:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOC_STORE_PATH):
        raise FileNotFoundError("RAG index or doc store not found. Please run build_index() first.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOC_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    query_emb = encoder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return [documents[i] for i in I[0] if i < len(documents)]
