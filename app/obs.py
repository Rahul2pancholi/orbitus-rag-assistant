"""Structured JSON logging. On Lambda, stdout goes to CloudWatch Logs, where these
fields are queryable with Logs Insights (e.g. `stats avg(llm_ms) by bin(5m)`)."""

import json
import logging
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger("rag")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str))


@contextmanager
def timed(timings: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round((time.perf_counter() - start) * 1000, 1)
