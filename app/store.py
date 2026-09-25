"""Per-tenant vector index, persisted as one JSON object per tenant.

Layout (identical for a local directory and for S3):
    <store_uri>/tenants/<tenant_id>/index.json      chunks + vectors + doc hashes
    <store_uri>/tenants/<tenant_id>/docs/<file>     original documents (S3 only)

Isolation boundary: an index object only ever holds one tenant's chunks, and
search runs over a single loaded index, so a query cannot return another
tenant's content even if a filter is forgotten. In AWS the tenant prefix is also
the IAM boundary (the Lambda role is only granted s3:GetObject on its prefix).

Brute-force cosine search over an in-memory list is deliberate: it is exact and
fast for thousands of chunks. See README "Scaling" for the path to S3 Vectors /
OpenSearch when an index outgrows memory.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import TENANT_ID_RE, Settings

INDEX_CACHE_TTL_S = 60


class IndexNotFoundError(Exception):
    pass


@dataclass
class Record:
    id: str
    source: str
    page: int
    text: str
    vector: list[float]


@dataclass
class VectorIndex:
    tenant_id: str
    embed_model_id: str
    doc_hashes: dict[str, str] = field(default_factory=dict)  # source -> sha256 of file bytes
    records: list[Record] = field(default_factory=list)

    def replace_document(self, source: str, sha: str, records: list[Record]) -> None:
        self.records = [r for r in self.records if r.source != source] + records
        self.doc_hashes[source] = sha

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[Record, float]]:
        scored = [(r, sum(a * b for a, b in zip(query_vec, r.vector))) for r in self.records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "tenant_id": self.tenant_id,
                "embed_model_id": self.embed_model_id,
                "doc_hashes": self.doc_hashes,
                "records": [r.__dict__ for r in self.records],
            }
        ).encode()

    @classmethod
    def from_json(cls, raw: bytes) -> "VectorIndex":
        data = json.loads(raw)
        return cls(
            tenant_id=data["tenant_id"],
            embed_model_id=data["embed_model_id"],
            doc_hashes=data["doc_hashes"],
            records=[Record(**r) for r in data["records"]],
        )


class IndexStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        uri = settings.store_uri
        if uri.startswith("s3://"):
            import boto3

            bucket, _, prefix = uri[5:].partition("/")
            self._bucket, self._prefix = bucket, prefix.strip("/")
            self._s3 = boto3.client("s3", region_name=settings.aws_region)
        else:
            self._bucket, self._root = None, Path(uri)
        self._cache: dict[str, tuple[float, VectorIndex]] = {}

    def _key(self, tenant_id: str, *parts: str) -> str:
        if not TENANT_ID_RE.match(tenant_id):  # guards against path traversal via tenant id
            raise ValueError("invalid tenant id")
        return "/".join(p for p in (self._prefix if self._bucket else "", "tenants", tenant_id, *parts) if p)

    def _read(self, key: str) -> bytes:
        if self._bucket:
            try:
                return self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()
            except self._s3.exceptions.NoSuchKey as exc:
                raise IndexNotFoundError(key) from exc
        path = self._root / key
        if not path.exists():
            raise IndexNotFoundError(key)
        return path.read_bytes()

    def _write(self, key: str, data: bytes) -> None:
        if self._bucket:
            self._s3.put_object(Bucket=self._bucket, Key=key, Body=data)
        else:
            path = self._root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)  # atomic swap: readers never see a half-written index

    def load(self, tenant_id: str, use_cache: bool = False) -> VectorIndex:
        cached = self._cache.get(tenant_id)
        if use_cache and cached and time.monotonic() - cached[0] < INDEX_CACHE_TTL_S:
            return cached[1]
        index = VectorIndex.from_json(self._read(self._key(tenant_id, "index.json")))
        if index.embed_model_id != self._settings.embed_model_id:
            # Vectors from different models are not comparable; fail loudly instead of returning garbage.
            raise ValueError(
                f"index was built with {index.embed_model_id}, runtime uses "
                f"{self._settings.embed_model_id}; re-run ingestion"
            )
        self._cache[tenant_id] = (time.monotonic(), index)
        return index

    def load_or_create(self, tenant_id: str) -> VectorIndex:
        try:
            return self.load(tenant_id)
        except IndexNotFoundError:
            return VectorIndex(tenant_id=tenant_id, embed_model_id=self._settings.embed_model_id)

    def save(self, index: VectorIndex) -> None:
        self._write(self._key(index.tenant_id, "index.json"), index.to_json())
        self._cache[index.tenant_id] = (time.monotonic(), index)  # new docs are queryable immediately

    def save_document(self, tenant_id: str, name: str, data: bytes) -> None:
        if self._bucket:  # locally the originals already live on disk
            self._write(self._key(tenant_id, "docs", Path(name).name), data)
