"""
backend/main.py
FastAPI Web Server for NovaCart Essentials Customer Support RAG System.
Features:
- Hybrid Search (BM25 + Dense Sentence Transformers + RRF)
- Cross-Encoder Reranking
- Deterministic Redis Answer Caching
- Real-time Performance Telemetry & Policy Invalidation Endpoints
"""

import os
from pathlib import Path

# Ensure fast offline model loading without HuggingFace DNS retry hangs
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.rag.pipeline import CustomerSupportRAGPipeline, SupportRAGResponse
from backend.rag.retrieval import KnowledgeBaseManager
from backend.cache.redis_cache import CustomerSupportCache

app = FastAPI(
    title="NovaCart Customer Support RAG",
    description="Customer Support RAG with BM25 + Dense Hybrid Search, Cross-Encoder Reranking, and Redis Caching.",
    version="2.0.0",
)

# Enable CORS for Next.js and frontend dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline once at application startup
pipeline = CustomerSupportRAGPipeline()

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public"


class ChatRequest(BaseModel):
    question: str
    skip_cache: bool = False
    use_reranker: bool = True


class UpdatePolicyRequest(BaseModel):
    chunk_id: str
    new_policy_text: str


@app.get("/api/health")
async def health_check():
    """System health and operational status."""
    return {
        "status": "healthy",
        "company": "NovaCart Essentials",
        "kb_version": pipeline.kb_manager.version_hash,
        "total_chunks": len(pipeline.kb_manager.chunks),
        "redis_connected": pipeline.cache.is_available,
        "is_emulator": pipeline.cache._is_emulator,
        "llm_model": pipeline.generator.model,
    }


@app.post("/api/chat", response_model=Dict[str, Any])
async def handle_customer_chat(request: ChatRequest):
    """
    Primary Customer Support Chat Endpoint.
    Executes the exact workflow:
    1. Redis Lookup -> On HIT, returns in <1ms without calling retrieval or LLM.
    2. On MISS -> BM25 + Dense -> RRF -> Cross-Encoder -> LLM -> Cache -> return.
    """
    q = request.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response: SupportRAGResponse = pipeline.answer_question(
            question=q,
            skip_cache=request.skip_cache,
            use_reranker=request.use_reranker,
        )
        return response.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/api/kb")
async def get_knowledge_base():
    """Return knowledge base documents and version fingerprint."""
    docs_summary = []
    seen_docs = set()
    for c in pipeline.kb_manager.chunks:
        if c.doc_id not in seen_docs:
            seen_docs.add(c.doc_id)
            docs_summary.append({
                "doc_id": c.doc_id,
                "title": c.title,
                "version": c.doc_version,
                "source_ref": c.source_ref,
            })

    return {
        "company": "NovaCart Essentials",
        "version_hash": pipeline.kb_manager.version_hash,
        "total_chunks": len(pipeline.kb_manager.chunks),
        "documents": docs_summary,
        "chunks": [c.to_dict() for c in pipeline.kb_manager.chunks],
    }


@app.get("/api/metrics")
async def get_cache_metrics():
    """Return live tracked performance telemetry for the customer support dashboard."""
    return pipeline.cache.get_metrics()


@app.post("/api/cache/clear")
async def clear_cache():
    """Flush all cached customer support answers."""
    success = pipeline.cache.clear()
    return {"message": "Redis answer cache successfully cleared.", "success": success}


@app.post("/api/kb/update-policy")
async def update_sample_policy(request: UpdatePolicyRequest):
    """
    Demonstrates Section 8 Cache Invalidation:
    Updates a sample policy chunk and re-computes the knowledge-base version hash.
    Past cached answers are immediately bypassed without returning stale data.
    """
    try:
        new_version = pipeline.kb_manager.update_document_chunk(
            chunk_id=request.chunk_id,
            new_text=request.new_policy_text,
        )
        pipeline.reload_knowledge_base()
        return {
            "message": f"Successfully updated chunk '{request.chunk_id}'.",
            "old_version": pipeline.kb_manager.version_hash,
            "new_version_hash": new_version,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Serve static web frontend if built
if STATIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(STATIC_DIR)), name="public")

    @app.get("/")
    async def serve_frontend():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "NovaCart Customer Support RAG API is operational."}

