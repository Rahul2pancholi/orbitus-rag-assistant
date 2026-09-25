"""Embedding providers. Both return L2-normalised vectors so cosine == dot product."""

import json
import math
import os
from functools import lru_cache
from typing import Protocol

from app.config import Settings


class Embedder(Protocol):
    model_id: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class LocalEmbedder:
    """On-device ONNX model via fastembed (~70 MB download on first use).

    On Lambda the model files are bundled in the package and EMBED_MODEL_PATH points at them.
    """

    def __init__(self, model_id: str):
        from fastembed import TextEmbedding

        self.model_id = model_id
        self._model = TextEmbedding(model_id, specific_model_path=os.environ.get("EMBED_MODEL_PATH"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_normalise(v.tolist()) for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _normalise(next(iter(self._model.query_embed(text))).tolist())


class BedrockEmbedder:
    """Amazon Titan Text Embeddings v2. The API takes one text per call."""

    DIMENSIONS = 512  # 256/512/1024 supported; 512 keeps ~99% of retrieval quality at half the storage

    def __init__(self, model_id: str, region: str):
        import boto3
        from botocore.config import Config

        self.model_id = model_id
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(read_timeout=20, retries={"max_attempts": 4, "mode": "adaptive"}),
        )

    def _embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text, "dimensions": self.DIMENSIONS, "normalize": True})
        response = self._client.invoke_model(modelId=self.model_id, body=body)
        return json.loads(response["body"].read())["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@lru_cache(maxsize=1)
def get_embedder(settings: Settings) -> Embedder:
    if settings.embeddings_provider == "bedrock":
        return BedrockEmbedder(settings.embed_model_id, settings.aws_region)
    return LocalEmbedder(settings.embed_model_id)
