from rag_retriever import build_index

# 🔹 Your knowledge base: add your custom text chunks here
docs = [
    "TinyLLaMA is a lightweight open-source model.",
    "RAG enables LLMs to retrieve external knowledge.",
    "Streamlit is a fast way to build AI web apps.",
    "FAISS is used for fast vector similarity search.",
    "SentenceTransformers allow semantic embedding generation."
]

# 🔧 Build the RAG index (creates rag_index.faiss and rag_docs.pkl)
build_index(docs)

print("✅ RAG index and doc store built.")
