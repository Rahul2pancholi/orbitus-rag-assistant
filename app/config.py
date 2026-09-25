"""All runtime configuration comes from environment variables.

Local mode (default) needs no AWS at all. AWS mode is selected purely by env vars,
which the CDK stack sets on the Lambda function. See .env.example.
"""

import os
import re
from dataclasses import dataclass

TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")

# Cosine-similarity floor below which a chunk is treated as "not evidence".
# It is deliberately a LOW floor: it only skips the LLM for clearly unrelated
# questions. Scores of relevant vs. off-topic questions overlap on real documents
# (e.g. a vague "what do you know about nda" scored 0.55, "parental leave" 0.64),
# so the final relevance call is the LLM's NOT_FOUND gate, not this number.
# Score distributions differ per embedding model, so the default is per provider;
# the Bedrock value is a starting point to re-calibrate once Bedrock access exists.
DEFAULT_MIN_SCORE = {"local": 0.5, "bedrock": 0.25}

DEFAULT_LLM_MODEL = {
    "extractive": "",
    "ollama": "qwen2.5:3b",
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "groq": "openai/gpt-oss-120b",
}


@dataclass(frozen=True)
class Settings:
    embeddings_provider: str  # "local" (fastembed) | "bedrock" (Titan v2)
    llm_provider: str  # "extractive" (no LLM) | "ollama" (local model) | "bedrock" (Claude via Converse) | "groq" (hosted API)
    store_uri: str  # local directory, or s3://bucket[/prefix]
    tenant_id: str
    aws_region: str
    embed_model_id: str
    llm_model_id: str
    top_k: int
    min_score: float
    chunk_size: int
    chunk_overlap: int
    llm_timeout_s: int
    ollama_url: str
    groq_api_key_param: str  # SSM SecureString name holding the Groq key (AWS); locally GROQ_API_KEY is used


def load_settings() -> Settings:
    env = os.environ.get
    embeddings_provider = env("EMBEDDINGS_PROVIDER", "local")
    llm_provider = env("LLM_PROVIDER", "extractive")
    tenant_id = env("TENANT_ID", "demo")

    if embeddings_provider not in DEFAULT_MIN_SCORE:
        raise ValueError(f"EMBEDDINGS_PROVIDER must be one of {list(DEFAULT_MIN_SCORE)}")
    if llm_provider not in DEFAULT_LLM_MODEL:
        raise ValueError(f"LLM_PROVIDER must be one of {list(DEFAULT_LLM_MODEL)}")
    if not TENANT_ID_RE.match(tenant_id):
        raise ValueError("TENANT_ID must be lowercase alphanumeric/hyphen")

    default_embed_model = (
        "amazon.titan-embed-text-v2:0" if embeddings_provider == "bedrock" else "BAAI/bge-small-en-v1.5"
    )
    return Settings(
        embeddings_provider=embeddings_provider,
        llm_provider=llm_provider,
        store_uri=env("STORE_URI", ".data"),
        tenant_id=tenant_id,
        aws_region=env("AWS_REGION", "us-east-2"),
        embed_model_id=env("EMBED_MODEL_ID", default_embed_model),
        llm_model_id=env("LLM_MODEL_ID", DEFAULT_LLM_MODEL[llm_provider]),
        top_k=int(env("TOP_K", "5")),
        min_score=float(env("MIN_SCORE", DEFAULT_MIN_SCORE[embeddings_provider])),
        chunk_size=int(env("CHUNK_SIZE", "600")),
        chunk_overlap=int(env("CHUNK_OVERLAP", "120")),
        # a local model on a laptop is much slower than Bedrock, especially on first load
        llm_timeout_s=int(env("LLM_TIMEOUT_S", "120" if llm_provider == "ollama" else "30")),
        ollama_url=env("OLLAMA_URL", "http://localhost:11434"),
        groq_api_key_param=env("GROQ_API_KEY_PARAM", ""),
    )
