import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from backend.rag.pipeline import CustomerSupportRAGPipeline
from backend.cache.redis_cache import CustomerSupportCache, RedisUnavailable
from backend.rag.generator import CustomerSupportGenerator
from backend.admission import RequestLimiter, SupportBusy, positive_setting
from backend.locking import single_worker

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public"

@asynccontextmanager
async def lifespan(app):
    generator = CustomerSupportGenerator()
    generator.validate_configuration()
    support_contact_url()
    app.state.limiter = RequestLimiter(limit=positive_setting("CHAT_REQUESTS_PER_MINUTE", 20))
    # Multiple processes with independent policy revisions must not serve this store.
    with single_worker(Path(__file__).parent / "data" / ".server.lock"):
        cache = CustomerSupportCache()
        try:
            app.state.pipeline = CustomerSupportRAGPipeline(cache=cache, generator=generator)
            yield
        finally:
            cache.close()

app = FastAPI(title="NovaCart Support", version="2.1.0", lifespan=lifespan)

def get_pipeline(request: Request):
    return request.app.state.pipeline

def get_limiter(request: Request):
    return request.app.state.limiter

def support_contact_url():
    value = os.getenv("SUPPORT_CONTACT_URL", "").strip()
    if value:
        url = urlsplit(value)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("SUPPORT_CONTACT_URL must be an HTTPS support page.")
    return value

def verify_admin_access(x_admin_token: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    expected = os.getenv("ADMIN_API_KEY", "")
    if not expected or expected == "novacart-admin-key-2026" or expected.startswith("your_"):
        raise HTTPException(503, "Administrative actions are disabled. Configure a private ADMIN_API_KEY.")
    token = x_admin_token or (authorization[7:] if authorization and authorization.startswith("Bearer ") else "")
    if not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(401, "Valid admin credentials required.")

class MetadataFilters(BaseModel):
    model_config = {"extra": "forbid"}
    doc_id: Optional[str] = None
    section: Optional[str] = None
    doc_version: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    skip_cache: bool = False
    use_reranker: bool = True
    metadata_filters: Optional[MetadataFilters] = None

class UpdatePolicyRequest(BaseModel):
    chunk_id: str = Field(min_length=1, max_length=200)
    new_policy_text: str = Field(min_length=1, max_length=20000)

@app.get("/api/health")
def health_check(pipeline=Depends(get_pipeline)):
    connected = pipeline.cache.is_available
    configured = bool(pipeline.generator.configured)
    provider_error = pipeline.generator.last_error
    ready = configured and not provider_error
    reranker_available = bool(pipeline.reranker.is_available)
    return JSONResponse(status_code=200 if ready else 503, content={"status": ("ready" if connected and reranker_available else "degraded") if ready else "unavailable", "kb_version": pipeline.kb_manager.version_hash,
            "total_chunks": len(pipeline.kb_manager.chunks), "redis_connected": connected,
            "cache_mode": "redis" if connected else "uncached",
            "llm_configured": configured, "provider_status": "error" if provider_error else "reachable" if pipeline.generator.provider_contacted else "not_probed",
            "reranker_available": reranker_available,
            "generation_mode": pipeline.generator.model})

@app.post("/api/chat")
def handle_customer_chat(request: ChatRequest, http_request: Request, pipeline=Depends(get_pipeline),
                         limiter=Depends(get_limiter), x_admin_token: Optional[str] = Header(None),
                         authorization: Optional[str] = Header(None)):
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")
    if request.skip_cache:
        verify_admin_access(x_admin_token, authorization)
    client = http_request.client.host if http_request.client else "unknown"
    if not limiter.allow(client):
        raise HTTPException(429, "Too many questions. Please retry in a minute.", headers={"Retry-After": "60"})
    try:
        filters = request.metadata_filters.model_dump(exclude_none=True) if request.metadata_filters else None
        result = pipeline.answer_question(question, skip_cache=request.skip_cache, use_reranker=request.use_reranker,
                                          metadata_filters=filters)
        payload = result.to_dict()
        payload["support_contact_url"] = support_contact_url()
        return JSONResponse(status_code=503 if result.error else 200, content=payload)
    except SupportBusy as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "5"}) from None
    except RedisUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        logger.exception("Question processing failed")
        raise HTTPException(503, "Support is temporarily unavailable. Please try again.")

@app.get("/api/kb")
def get_knowledge_base(pipeline=Depends(get_pipeline)):
    try:
        return pipeline.get_knowledge_base()
    except RedisUnavailable as exc:
        raise HTTPException(503, str(exc)) from None

@app.get("/api/metrics")
def get_cache_metrics(pipeline=Depends(get_pipeline)):
    return pipeline.cache.get_metrics()

@app.post("/api/cache/clear", dependencies=[Depends(verify_admin_access)])
def clear_cache(pipeline=Depends(get_pipeline)):
    if not pipeline.cache.clear():
        raise HTTPException(503, "Cache is unavailable.")
    return {"success": True}

@app.post("/api/kb/update-policy", dependencies=[Depends(verify_admin_access)])
def update_policy(request: UpdatePolicyRequest, pipeline=Depends(get_pipeline)):
    try:
        return pipeline.stage_and_update_policy(request.chunk_id, request.new_policy_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Policy update failed")
        raise HTTPException(503, "Policy update failed. Please retry.")

app.mount("/public", StaticFiles(directory=STATIC_DIR), name="public")

@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")
