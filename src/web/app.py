"""
src/web/app.py
FastAPI web server for the PDF-based RAG application.
Provides REST API endpoints for:
- Document upload and ingestion
- Knowledge base management and deletion
- Hybrid semantic question-answering with citation retrieval
- Real-time system health and index metrics
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import config
from src.rag_pipeline import RAGPipeline, RAGAnswer


# Initialize FastAPI app
app = FastAPI(
    title="Minimalist PDF RAG",
    description="Fast, minimalist, beautiful web interface for local PDF question-answering.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Shared singleton RAG pipeline
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = None


class CacheStrategyRequest(BaseModel):
    strategy: str


@app.get("/")
async def serve_index():
    """Serve the single-page application frontend."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/api/status")
async def get_system_status():
    """Return live system health, LLM provider, vector DB metrics, and Redis cache stats."""
    llm_connected = pipeline.generator.check_connection()
    doc_summaries = pipeline.list_documents()

    return {
        "status": "online",
        "llm_provider": pipeline.generator.provider,
        "llm_model": pipeline.generator.model_name,
        "llm_connected": llm_connected,
        "embedding_model": config.embedding_model_name,
        "embedding_dimension": 384,
        "relevance_threshold": config.relevance_threshold,
        "total_records": pipeline.vector_store.count(),
        "total_documents": len(doc_summaries),
        "cache": pipeline.cache_store.get_metrics(),
        "cache_strategy": pipeline.cache_strategy.name,
    }


@app.get("/api/documents")
async def list_documents():
    """Return all currently indexed documents in ChromaDB."""
    documents = pipeline.list_documents()
    return {"documents": documents, "count": len(documents)}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more PDF files, save to documents folder, and index them."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided.")

    saved_files = []
    saved_paths = []
    config.ensure_directories()

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: '{file.filename}'. Only .pdf files are supported."
            )

        target_path = config.documents_dir / file.filename
        try:
            with open(target_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file.filename)
            saved_paths.append(target_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file '{file.filename}': {str(e)}"
            )

    # Ingest newly uploaded files incrementally
    try:
        summary = pipeline.ingest_files(saved_paths, force_reindex=False)
        return {
            "message": f"Successfully uploaded and processed {len(saved_files)} file(s).",
            "uploaded_files": saved_files,
            "ingestion_summary": summary,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed during processing: {str(e)}"
        )


@app.post("/api/query")
async def ask_question(request: QueryRequest):
    """
    Answer user questions using hybrid retrieval and grounded generation.
    Supports multi-document scoping (document_ids).
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag_answer: RAGAnswer = pipeline.answer_question(
            question=question,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )

        return {
            "question": rag_answer.question,
            "answer": rag_answer.answer,
            "has_evidence": rag_answer.has_evidence,
            "sources": rag_answer.sources,
            "retrieved_passages": rag_answer.retrieved_passages,
            "model_used": rag_answer.model_used,
            "document_scope": rag_answer.document_scope,
            "collection_version": rag_answer.collection_version,
            "cache_key": rag_answer.cache_key,
            "error": rag_answer.error,
            "is_cached": rag_answer.is_cached,
            "latency_ms": rag_answer.latency_ms,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during question answering: {str(e)}"
        )


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Return real-time cache statistics, hit rate, and memory usage."""
    return {
        "strategy": pipeline.cache_strategy.name,
        "metrics": pipeline.cache_store.get_metrics(),
    }


@app.delete("/api/cache")
async def clear_cache():
    """Flush all cached RAG answers from Redis."""
    success = pipeline.clear_cache()
    pipeline.cache_store.reset_metrics()
    return {"message": "Cache successfully cleared.", "success": success}


@app.post("/api/cache/strategy")
async def set_cache_strategy(request: CacheStrategyRequest):
    """Switch the active caching strategy dynamically."""
    valid_strategies = ["cache_aside", "read_through", "write_through", "write_behind", "write_around"]
    strategy = request.strategy.lower().strip()
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid caching strategy '{request.strategy}'. Valid options: {valid_strategies}",
        )

    pipeline.set_cache_strategy(strategy)
    return {
        "message": f"Active caching strategy updated to '{strategy}'.",
        "current_strategy": pipeline.cache_strategy.name,
    }


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from both the vector store and the documents folder."""
    deleted_chunks = pipeline.vector_store.delete_document(filename)

    file_path = config.documents_dir / filename
    file_deleted = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_deleted = True
        except Exception:
            pass

    return {
        "message": f"Successfully removed '{filename}'.",
        "filename": filename,
        "deleted_chunks": deleted_chunks,
        "file_deleted": file_deleted,
        "remaining_records": pipeline.vector_store.count(),
    }


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

