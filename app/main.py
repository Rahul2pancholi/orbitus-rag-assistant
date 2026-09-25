"""FastAPI app. Runs under uvicorn locally and under Lambda via Mangum (`handler`)."""

import logging
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from mangum import Mangum
from pydantic import BaseModel, Field, field_validator

from app.chunking import SUPPORTED_SUFFIXES
from app.config import load_settings
from app.ingest import ingest_paths
from app.obs import log_event
from app.rag import answer_question
from app.store import IndexNotFoundError, IndexStore

settings = load_settings()
store = IndexStore(settings)
app = FastAPI(title="Orbitus RAG Operations Assistant")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Serialises index writes within this process (read-modify-write of index.json).
# Across processes/Lambdas this needs a real lock; see README limitations.
_ingest_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

    @field_validator("question")
    @classmethod
    def strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


def resolve_tenant(request: Request) -> str:
    """IMPLEMENTED: single tenant from server config.
    PROPOSED: derive from a verified auth token claim (e.g. Cognito `custom:tenant_id`).
    The tenant is deliberately never read from the request body or query string."""
    return settings.tenant_id


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log_event("unhandled_error", level=logging.ERROR, request_id=request_id, path=request.url.path)
        response = JSONResponse({"error": "internal error", "request_id": request_id}, status_code=500)
    response.headers["x-request-id"] = request_id
    log_event("http", request_id=request_id, method=request.method, path=request.url.path,
              status=response.status_code, total_ms=round((time.perf_counter() - start) * 1000, 1))
    return response


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "embeddings": settings.embeddings_provider, "llm": settings.llm_provider}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    tenant_id = resolve_tenant(request)
    try:
        return answer_question(body.question, tenant_id, settings, store)
    except IndexNotFoundError:
        return JSONResponse({"error": "No documents have been ingested yet. Run the ingestion script."},
                            status_code=503)
    except ValueError as exc:  # e.g. embedding model mismatch between index and runtime
        log_event("config_error", level=logging.ERROR, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
    except Exception as exc:  # embedding service / storage failures
        log_event("ask_failed", level=logging.ERROR, error=repr(exc))
        return JSONResponse({"error": "Retrieval service is temporarily unavailable. Please retry."},
                            status_code=503)


@app.post("/api/documents")
def upload_document(file: UploadFile, request: Request):
    tenant_id = resolve_tenant(request)
    name = Path(file.filename or "").name  # drop any client-supplied directories
    if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
        return JSONResponse({"error": "Only PDF, TXT and MD files are supported."}, status_code=415)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "File is larger than 10 MB."}, status_code=413)
    if not data:
        return JSONResponse({"error": "File is empty."}, status_code=400)

    with tempfile.TemporaryDirectory() as tmp, _ingest_lock:
        path = Path(tmp) / name
        path.write_bytes(data)
        report = ingest_paths([path], settings, tenant_id=tenant_id, store=store)
    if report["failed"]:
        return JSONResponse({"error": report["failed"][0]["error"]}, status_code=422)
    return report


handler = Mangum(app, lifespan="off")
