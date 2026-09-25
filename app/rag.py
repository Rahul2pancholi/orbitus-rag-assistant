"""Retrieve-then-generate pipeline with explicit handling of missing evidence and LLM failure."""

import logging
import re

from app.config import Settings
from app.embeddings import get_embedder
from app.llm import NOT_FOUND_TOKEN, ExtractiveLLM, LLMError, get_llm
from app.obs import log_event, timed
from app.store import IndexStore

NOT_FOUND_MESSAGE = "I couldn't find this in the supplied documents."


def answer_question(question: str, tenant_id: str, settings: Settings, store: IndexStore) -> dict:
    timings: dict = {}
    index = store.load(tenant_id, use_cache=True)

    with timed(timings, "embed_ms"):
        query_vec = get_embedder(settings).embed_query(question)
    with timed(timings, "search_ms"):
        hits = index.search(query_vec, settings.top_k)

    evidence = [(r, s) for r, s in hits if s >= settings.min_score]
    retrieved = [
        {"id": r.id, "source": r.source, "page": r.page, "score": round(s, 3), "text": r.text,
         "used": s >= settings.min_score}
        for r, s in hits
    ]
    result = {"question": question, "retrieved": retrieved, "status": "answered", "usage": {}}
    log_fields = {
        "tenant_id": tenant_id,
        "top_scores": [c["score"] for c in retrieved],
        "evidence_count": len(evidence),
    }

    # Gate 1: retrieval found nothing relevant -> refuse without spending an LLM call.
    if not evidence:
        result.update(status="not_found", answer=NOT_FOUND_MESSAGE, sources=[])
        log_event("ask", status="not_found", **log_fields, **timings)
        return result

    passages = [r for r, _ in evidence]
    try:
        with timed(timings, "llm_ms"):
            text, usage = get_llm(settings).generate(question, passages)
    except LLMError as exc:
        # Degrade instead of failing: show the evidence we found and say the model is unavailable.
        text, usage = ExtractiveLLM().generate(question, passages)
        text = "The language model is currently unavailable, showing the best matching passages instead.\n\n" + text
        result["status"] = "degraded"
        log_event("llm_error", level=logging.ERROR, error=str(exc), tenant_id=tenant_id)

    text = text.replace("【", "[").replace("】", "]")  # gpt-oss cites with CJK brackets; normalise for UI + parsing

    # Gate 2: the model read the evidence and judged it insufficient.
    if text.strip() == NOT_FOUND_TOKEN:
        result.update(status="not_found", answer=NOT_FOUND_MESSAGE, sources=[])
    else:
        result.update(answer=text, usage=usage, sources=_sources(_cited(text, passages)))

    log_event("ask", status=result["status"], **log_fields, **usage, **timings)
    return result


def _cited(text: str, passages: list) -> list:
    """Passages the answer actually cites as [n]; all passages if it cites none."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    cited = [p for i, p in enumerate(passages, 1) if i in numbers]
    return cited or passages


def _sources(passages) -> list[dict]:
    """Unique documents, in citation order, with the pages that were used."""
    by_source: dict[str, set[int]] = {}
    for p in passages:
        by_source.setdefault(p.source, set()).add(p.page)
    return [{"source": s, "pages": sorted(pages)} for s, pages in by_source.items()]
