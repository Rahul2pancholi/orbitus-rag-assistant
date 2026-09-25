"""Ingestion: file -> pages -> chunks -> embeddings -> tenant index.

Idempotency: each file is keyed by name and fingerprinted by the sha256 of its
bytes. Unchanged files are skipped (no embedding cost); changed files have all
their old chunks replaced, so re-running ingestion never creates duplicates.
Chunk ids are content hashes, so they are stable across runs.
"""

import hashlib
from pathlib import Path

from app.chunking import SUPPORTED_SUFFIXES, ExtractionError, chunk_pages, extract_pages
from app.config import Settings
from app.embeddings import get_embedder
from app.obs import log_event, timed
from app.store import IndexStore, Record


def ingest_paths(
    paths: list[Path], settings: Settings, tenant_id: str | None = None, store: IndexStore | None = None
) -> dict:
    tenant_id = tenant_id or settings.tenant_id
    store = store or IndexStore(settings)
    index = store.load_or_create(tenant_id)
    embedder = get_embedder(settings)
    report = {"tenant_id": tenant_id, "ingested": [], "skipped": [], "failed": []}

    files = sorted(p for p in paths if p.suffix.lower() in SUPPORTED_SUFFIXES)
    for path in files:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if index.doc_hashes.get(path.name) == sha:
            report["skipped"].append(path.name)
            continue

        timings: dict = {}
        try:
            with timed(timings, "extract_ms"):
                chunks = chunk_pages(extract_pages(path), settings.chunk_size, settings.chunk_overlap)
            with timed(timings, "embed_ms"):
                vectors = embedder.embed_documents([c.text for c in chunks])
        except ExtractionError as exc:
            report["failed"].append({"file": path.name, "error": str(exc)})
            log_event("ingest_failed", file=path.name, error=str(exc))
            continue
        except Exception as exc:  # embedding service errors: keep going with other files
            report["failed"].append({"file": path.name, "error": f"embedding failed: {exc}"})
            log_event("ingest_failed", file=path.name, error=repr(exc))
            continue

        records = [
            Record(
                id=hashlib.sha256(f"{path.name}|{c.page}|{c.text}".encode()).hexdigest()[:16],
                source=path.name,
                page=c.page,
                text=c.text,
                vector=v,
            )
            for c, v in zip(chunks, vectors)
        ]
        index.replace_document(path.name, sha, records)
        store.save_document(tenant_id, path.name, data)
        report["ingested"].append({"file": path.name, "chunks": len(records)})
        log_event("ingest_file", tenant_id=tenant_id, file=path.name, chunks=len(records), **timings)

    # Save once at the end: the index is only ever replaced whole, so a crash mid-run
    # leaves the previous index intact and the next run redoes only the unsaved files.
    if report["ingested"]:
        store.save(index)
    report["total_chunks"] = len(index.records)
    return report
