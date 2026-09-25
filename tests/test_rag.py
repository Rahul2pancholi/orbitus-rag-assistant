"""Tests run fully offline with a deterministic fake embedder (bag-of-words hashing),
so they exercise chunking, ingestion, retrieval gates and error handling without models."""

import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from app import embeddings, llm, rag
from app.chunking import ExtractionError, chunk_pages, extract_pages
from app.config import load_settings
from app.ingest import ingest_paths
from app.store import IndexNotFoundError, IndexStore


class FakeEmbedder:
    model_id = "fake-bow"

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * 256
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FailingLLM:
    def generate(self, question, passages):
        raise llm.LLMError("simulated timeout")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "get_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(rag, "get_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr("app.ingest.get_embedder", lambda s: FakeEmbedder())
    base = load_settings()
    return replace(base, store_uri=str(tmp_path / "store"), embed_model_id="fake-bow", min_score=0.3)


@pytest.fixture
def docs(tmp_path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "keys.txt").write_text("Access keys must be rotated every 90 days. Secrets live in Secrets Manager.")
    (d / "deploy.md").write_text("Production deploys are not allowed on Friday afternoon or weekends.")
    return d


def test_chunking_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_pages([(1, text)], chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 230 for c in chunks)
    # the last sentence of chunk N reappears at the start of chunk N+1
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_empty_and_unsupported_files_fail_cleanly(tmp_path):
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "x.docx").write_bytes(b"nope")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    for name in ("empty.txt", "x.docx", "bad.pdf"):
        with pytest.raises(ExtractionError):
            extract_pages(tmp_path / name)


def test_ingestion_is_idempotent_and_replaces_changed_docs(settings, docs):
    first = ingest_paths(list(docs.iterdir()), settings)
    assert len(first["ingested"]) == 2
    total = first["total_chunks"]

    second = ingest_paths(list(docs.iterdir()), settings)
    assert second["ingested"] == [] and len(second["skipped"]) == 2
    assert second["total_chunks"] == total

    (docs / "keys.txt").write_text("Access keys must be rotated every 30 days.")
    third = ingest_paths(list(docs.iterdir()), settings)
    assert [i["file"] for i in third["ingested"]] == ["keys.txt"]
    texts = [r.text for r in IndexStore(settings).load("demo").records]
    assert not any("90 days" in t for t in texts)


def test_bad_file_does_not_abort_batch(settings, docs):
    (docs / "broken.pdf").write_bytes(b"garbage")
    report = ingest_paths(list(docs.iterdir()), settings)
    assert len(report["ingested"]) == 2
    assert report["failed"][0]["file"] == "broken.pdf"


def test_answer_with_sources(settings, docs):
    ingest_paths(list(docs.iterdir()), settings)
    result = rag.answer_question("How often are access keys rotated?", "demo", settings, IndexStore(settings))
    assert result["status"] == "answered"
    assert result["sources"][0]["source"] == "keys.txt"
    assert "90 days" in result["answer"]


def test_unrelated_question_is_not_found_without_llm_call(settings, docs, monkeypatch):
    ingest_paths(list(docs.iterdir()), settings)
    monkeypatch.setattr(rag, "get_llm", lambda s: pytest.fail("LLM must not be called"))
    result = rag.answer_question("What is the parental leave policy?", "demo", settings, IndexStore(settings))
    assert result["status"] == "not_found"
    assert result["sources"] == []


def test_llm_not_found_token_is_honoured(settings, docs, monkeypatch):
    ingest_paths(list(docs.iterdir()), settings)

    class NotFoundLLM:
        def generate(self, q, p):
            return llm.NOT_FOUND_TOKEN, {}

    monkeypatch.setattr(rag, "get_llm", lambda s: NotFoundLLM())
    result = rag.answer_question("Are access keys rotated?", "demo", settings, IndexStore(settings))
    assert result["status"] == "not_found"


def test_llm_failure_degrades_to_extractive(settings, docs, monkeypatch):
    ingest_paths(list(docs.iterdir()), settings)
    monkeypatch.setattr(rag, "get_llm", lambda s: FailingLLM())
    result = rag.answer_question("How often are access keys rotated?", "demo", settings, IndexStore(settings))
    assert result["status"] == "degraded"
    assert "unavailable" in result["answer"] and result["sources"]


def test_tenants_are_isolated(settings, docs, tmp_path):
    ingest_paths([docs / "keys.txt"], settings, tenant_id="acme")
    other = tmp_path / "other"
    other.mkdir()
    (other / "globex.txt").write_text("Globex secret project codename is Nightjar.")
    ingest_paths([other / "globex.txt"], settings, tenant_id="globex")

    store = IndexStore(settings)
    result = rag.answer_question("What is the secret project codename?", "acme", settings, store)
    assert all(c["source"] != "globex.txt" for c in result["retrieved"])
    with pytest.raises(IndexNotFoundError):
        store.load("initech")
    with pytest.raises(ValueError):
        store.load("../globex")


def test_api_validation_and_missing_index(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main, "store", IndexStore(settings))
    client = TestClient(main.app)
    assert client.post("/api/ask", json={"question": "  "}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422
    resp = client.post("/api/ask", json={"question": "anything at all?"})
    assert resp.status_code == 503 and "ingested" in resp.json()["error"]


def test_sources_are_limited_to_cited_passages(settings, docs, monkeypatch):
    ingest_paths(list(docs.iterdir()), settings)

    class CitingLLM:
        def generate(self, q, passages):
            return "Rotate every 90 days [1].", {}

    monkeypatch.setattr(rag, "get_llm", lambda s: CitingLLM())
    settings = replace(settings, min_score=0.0)
    result = rag.answer_question("access keys rotated deploys", "demo", settings, IndexStore(settings))
    assert len([c for c in result["retrieved"] if c["used"]]) == 2
    assert len(result["sources"]) == 1


def test_upload_endpoint(settings, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    monkeypatch.setattr(main, "store", IndexStore(settings))
    monkeypatch.setattr(main, "settings", settings)
    client = TestClient(main.app)

    ok = client.post("/api/documents", files={"file": ("../../leave.txt", b"Parental leave is 26 weeks.")})
    assert ok.status_code == 200 and ok.json()["ingested"][0]["file"] == "leave.txt"
    again = client.post("/api/documents", files={"file": ("leave.txt", b"Parental leave is 26 weeks.")})
    assert again.json()["skipped"] == ["leave.txt"]
    # uploaded content is queryable immediately (cache refreshed on save)
    answer = client.post("/api/ask", json={"question": "How long is parental leave?"}).json()
    assert answer["sources"][0]["source"] == "leave.txt"

    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("e.txt", b"")}).status_code == 400
    assert client.post("/api/documents", files={"file": ("bad.pdf", b"nope")}).status_code == 422
