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
import faiss
import re
from sentence_transformers import SentenceTransformer

def chunk_text(text, max_words=50):
    """Split a long string into chunks of ~max_words tokens"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []

    for sent in sentences:
        current_chunk.append(sent)
        if len(" ".join(current_chunk).split()) >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def build_index(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open("rag_docs.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    faiss.write_index(index, "rag_index.faiss")
    print(f"✅ RAG index built with {len(all_chunks)} chunks.")



def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOC_STORE_PATH):
        raise FileNotFoundError("RAG index or doc store not found. Please run build_index() first.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOC_STORE_PATH, "rb") as f:
        documents = pickle.load(f)

    query_emb = encoder.encode([query], convert_to_numpy=True, num_workers=0)
    D, I = index.search(query_emb, top_k)
    return [documents[i] for i in I[0] if i < len(documents)]