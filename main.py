 
# main.py
# PURPOSE: FastAPI application — exposes RAG pipeline as REST API
# WHY FastAPI: Fast, automatic docs, type validation, production ready
# INTERVIEW: "How did you deploy your RAG system?"

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

from rag.loader import load_pdf, get_document_info
from rag.chunker import chunk_documents
from rag.embeddings import create_vectorstore, load_vectorstore
from rag.retriever import retrieve_chunks, format_context
from rag.generator import generate_answer

# ── CREATE APP ──────────────────────────────────────────────
app = FastAPI(
    title="AI Document Intelligence API",
    description="Upload PDFs and ask questions using RAG",
    version="1.0.0"
)

# CORS middleware — allows frontend to call this API
# WHY: Without this, browsers block cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── PYDANTIC MODELS ─────────────────────────────────────────
# WHY Pydantic: Automatic validation + clear API contract
# If user sends wrong type → automatic 422 error with clear message

class QuestionRequest(BaseModel):
    question: str
    k: int = 3              # number of chunks to retrieve

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list
    tokens_used: int

class DocumentInfo(BaseModel):
    filename: str
    total_pages: int
    total_characters: int
    total_chunks: int
    status: str

# ── ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AI Document Intelligence API",
        "docs": "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health():
    """Check if API and vector store are ready"""
    has_db = os.path.exists("./chroma_db")
    return {
        "api": "healthy",
        "vector_store": "ready" if has_db else "empty - upload a PDF first"
    }


@app.post("/upload", response_model=DocumentInfo)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and process it for Q&A.
    
    Steps:
    1. Save PDF to uploads folder
    2. Load and extract text
    3. Split into chunks
    4. Create embeddings
    5. Store in ChromaDB
    
    WHY async: File uploads are I/O bound operations
    async allows other requests while file is uploading
    """
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Save uploaded file
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"\n📄 Processing: {file.filename}")
    
    # Run RAG pipeline
    pages = load_pdf(file_path)
    doc_info = get_document_info(pages)
    chunks = chunk_documents(pages)
    create_vectorstore(chunks)
    
    return DocumentInfo(
        filename=file.filename,
        total_pages=doc_info["total_pages"],
        total_characters=doc_info["total_characters"],
        total_chunks=len(chunks),
        status="ready"
    )


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded document.
    
    Steps:
    1. Load vector store
    2. Find relevant chunks
    3. Generate answer with sources
    """
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    # Load existing vector store
    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No document uploaded yet. POST to /upload first."
        )
    
    # Retrieve relevant chunks
    chunks = retrieve_chunks(vectorstore, request.question, k=request.k)
    context = format_context(chunks)
    
    # Generate answer
    result = generate_answer(request.question, context)
    
    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        sources=[{
            "page": c["page"],
            "filename": c["filename"],
            "similarity": c["similarity_score"],
            "preview": c["content"][:100] + "..."
        } for c in chunks],
        tokens_used=result["tokens_used"]
    )


@app.delete("/reset")
def reset_database():
    """Clear the vector store — useful for uploading new document"""
    import shutil
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        return {"status": "Vector store cleared. Upload a new PDF."}
    return {"status": "Nothing to clear."}


# ── RUN SERVER ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # reload=True → auto-restarts when you change code
    # Perfect for development