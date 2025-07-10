from rag_retriever import build_index

# 🔹 Your knowledge base: add your custom text chunks here
docs = [
    """TinyLLaMA is a lightweight open-source model designed to run on small devices. 
    It supports efficient inference while maintaining strong performance on language tasks. 
    This makes it ideal for edge deployment and mobile applications.""",

    """RAG enables LLMs to retrieve external knowledge at inference time. 
    This retrieval-augmented generation process allows grounding responses in a custom knowledge base, 
    improving factual accuracy and recall.""",

    ...
]

# 🔧 Build the RAG index (creates rag_index.faiss and rag_docs.pkl)
build_index(docs)

print("✅ RAG index and doc store built.")
