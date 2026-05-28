# AI Document Intelligence System

A production-ready RAG (Retrieval Augmented Generation) pipeline 
that answers questions from PDF documents with source citations.

## Features
- Upload any PDF and ask questions in natural language
- Returns answers with exact page number citations
- Semantic search using vector embeddings
- Zero hallucination — answers grounded in your documents
- REST API with interactive documentation

## Tech Stack
- **LangChain** — Document loading and text splitting
- **ChromaDB** — Vector database for semantic search
- **HuggingFace** — Free embedding model (all-MiniLM-L6-v2)
- **Groq + Llama 3** — Fast free LLM inference
- **FastAPI** — Production REST API

## Architecture
PDF → Chunks → Embeddings → ChromaDB → Semantic Search → LLM → Answer

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## API Endpoints
- `POST /upload` — Upload a PDF
- `POST /ask` — Ask a question
- `GET /health` — Check system status
- `DELETE /reset` — Clear database
