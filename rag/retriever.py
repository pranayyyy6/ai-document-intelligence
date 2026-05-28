 
# rag/retriever.py
# PURPOSE: Search vector store and return relevant chunks
# WHY: The "R" in RAG — Retrieval is the core of the system
# INTERVIEW: "How does retrieval work in your RAG system?"

def retrieve_chunks(vectorstore, query: str, k: int = 3) -> list:
    """
    Find k most relevant chunks for a query.
    
    WHY k=3?
    3 chunks = enough context for most questions
    Too many chunks = LLM gets confused, costs more tokens
    Too few chunks = might miss important information
    
    HOW IT WORKS:
    1. Query "What is machine learning?" → embed to vector
    2. Compare query vector to all stored vectors
    3. Return k chunks with highest cosine similarity
    
    COSINE SIMILARITY:
    Measures angle between two vectors (0 to 1)
    1.0 = identical meaning
    0.0 = completely unrelated
    """
    
    # similarity_search_with_score returns (document, score) pairs
    results_with_scores = vectorstore.similarity_search_with_score(
        query, k=k
    )
    
    chunks = []
    for doc, score in results_with_scores:
        chunks.append({
            "content": doc.page_content,
            "page": doc.metadata.get('page', 0) + 1,  # 0-indexed → 1-indexed
            "filename": doc.metadata.get('filename', 'unknown'),
            "similarity_score": round(float(score), 4)
            # Lower score = more similar in ChromaDB (distance metric)
        })
    
    return chunks


def format_context(chunks: list) -> str:
    """
    Format chunks into a clean context string for LLM.
    
    WHY format matters:
    LLM reads this as part of the prompt.
    Clear formatting = better answers.
    Including page numbers = source attribution.
    """
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} - Page {chunk['page']}]\n{chunk['content']}"
        )
    
    return "\n\n".join(context_parts)