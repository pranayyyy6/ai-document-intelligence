 
# rag/loader.py
# PURPOSE: Load PDF files and extract text with page numbers
# WHY SEPARATE FILE: Single responsibility — one file does one job
# INTERVIEW: "How do you load documents in RAG?" — this is your answer

from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path: str) -> list:
    """
    Load a PDF and return list of pages.
    
    Each page is a Document object with:
    - page_content: the actual text
    - metadata: {'page': 0, 'source': 'file.pdf'}
    
    WHY metadata matters:
    When user asks a question, we can say
    "Answer found on Page 3" — that's source citation
    """
    
    # Check file exists before loading
    # Production code always validates input
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    if not file_path.endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
    print(f"Loading PDF: {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"✅ Loaded {len(pages)} pages")
    
    # Add filename to metadata for better citations
    for page in pages:
        page.metadata['filename'] = os.path.basename(file_path)
    
    return pages


def get_document_info(pages: list) -> dict:
    """
    Return summary info about loaded document.
    WHY: Useful for API response and debugging
    """
    total_chars = sum(len(p.page_content) for p in pages)
    
    return {
        "total_pages": len(pages),
        "total_characters": total_chars,
        "filename": pages[0].metadata.get('filename', 'unknown'),
        "avg_chars_per_page": total_chars // len(pages)
    }