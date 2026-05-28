 
# rag/embeddings.py
# PURPOSE: Convert chunks to vectors + store in ChromaDB
# WHY: Enables semantic search — find by meaning not keywords
# INTERVIEW: "How do you store and search documents in RAG?"

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Global variable — embedding model loaded once, reused many times
# WHY: Loading model takes ~30 seconds, wasteful to reload every request
_embeddings_model = None

def get_embeddings_model():
    """
    Load embedding model once and cache it.
    
    WHY all-MiniLM-L6-v2?
    - Free, no API key needed
    - Fast: 384 dimensions (small but accurate)
    - Great for English text
    - Used by thousands of production systems
    
    Alternatives (for interviews):
    - OpenAI ada-002: better but costs money
    - BGE-large: better quality, slower
    - E5-large: multilingual support
    """
    global _embeddings_model
    
    if _embeddings_model is None:
        print("Loading embedding model (first time only)...")
        _embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # use GPU if available
            encode_kwargs={'normalize_embeddings': True}
            # normalize=True makes cosine similarity more accurate
        )
        print("✅ Embedding model loaded!")
    
    return _embeddings_model


def create_vectorstore(chunks: list, persist_dir: str = "./chroma_db"):
    """
    Create ChromaDB vectorstore from chunks.
    
    WHY persist_dir?
    Saves the database to disk so we don't
    re-embed documents every time app restarts.
    First run: slow (creates embeddings)
    Next runs: instant (loads from disk)
    
    INTERVIEW: "How do you handle persistence in RAG?"
    "We persist the vector store to disk using ChromaDB's
     built-in persistence. This means we only embed documents
     once and reload the index on subsequent requests."
    """
    embeddings = get_embeddings_model()
    
    print(f"Creating vector store with {len(chunks)} chunks...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"✅ Vector store created!")
    print(f"   Total vectors: {vectorstore._collection.count()}")
    print(f"   Saved to: {persist_dir}")
    
    return vectorstore


def load_vectorstore(persist_dir: str = "./chroma_db"):
    """
    Load existing vectorstore from disk.
    WHY: Avoid re-embedding on every app restart
    """
    embeddings = get_embeddings_model()
    
    if not os.path.exists(persist_dir):
        raise FileNotFoundError("No vector store found. Upload a PDF first.")
    
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    count = vectorstore._collection.count()
    print(f"✅ Loaded vector store: {count} vectors")
    
    return vectorstore