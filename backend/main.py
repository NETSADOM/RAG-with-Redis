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
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Header, Depends
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

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "novacart-admin-key-2026")


def verify_admin_access(
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Protect administrative endpoints from unauthorized access.
    Accepts X-Admin-Token or Bearer Authorization header.
    """
    token = x_admin_token
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if not token or token != ADMIN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid admin credentials required for this administrative operation.",
        )
    return True

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
        "eviction_policy": pipeline.cache.eviction_policy,
        "semantic_caching": pipeline.cache.semantic_caching_enabled,
        "llm_model": pipeline.generator.model,
    }


@app.post("/api/chat", response_model=Dict[str, Any])
@app.post("/api/query", response_model=Dict[str, Any])
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


@app.post("/api/cache/clear", dependencies=[Depends(verify_admin_access)])
async def clear_cache():
    """Flush NovaCart cached customer support answers."""
    success = pipeline.cache.clear()
    return {"message": "NovaCart answer cache successfully cleared.", "success": success}


@app.post("/api/kb/update-policy", dependencies=[Depends(verify_admin_access)])
async def update_sample_policy(request: UpdatePolicyRequest):
    """
    Safe, Retryable Policy Update & Revision Invalidation:
    1. Stages update and builds replacement indexes in memory first.
    2. Writes to disk only after index construction succeeds.
    3. Captures old_version and new_version to document what changed.
    4. Automatically invalidates old cache entries via revision bumping.
    """
    try:
        result = pipeline.stage_and_update_policy(
            chunk_id=request.chunk_id,
            new_text=request.new_policy_text,
        )

        return {
            "message": f"Successfully updated policy chunk '{request.chunk_id}'.",
            "doc_id": result["doc_id"],
            "chunk_id": result["chunk_id"],
            "old_version": result["old_version"],
            "new_version": result["new_version"],
            "old_text": result["old_text"],
            "new_text": result["new_text"],
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

