# Orbitus: AI/RAG Operations Assistant

Ask questions about a set of operations documents (PDF, TXT, MD) and get answers grounded in them, with sources, page numbers and the retrieved chunks. The app says clearly when the documents don't contain the answer.

- **Live URL:** https://a34k4d3w4erjh677qnqagsv6gi0dhaik.lambda-url.us-east-2.on.aws/
- **Stack:** Python 3.13 · FastAPI · AWS Lambda (Function URL) · S3 · SSM Parameter Store · on-Lambda embeddings (`BAAI/bge-small-en-v1.5`, ONNX) · Groq (`openai/gpt-oss-120b`) · AWS CDK. A Bedrock provider (Titan + Claude Haiku 4.5) is also implemented but can't be used on this AWS account (see Status).
- **Architecture diagram and isolation design:** [docs/architecture.md](docs/architecture.md)

## Status: what works today and what is proposed

| Capability | Status |
|---|---|
| Ingestion: extract text, chunk by sentence per page, embed, save a per-tenant index; re-runs are safe | ✅ Implemented and tested |
| Upload a PDF/TXT/MD from the UI (`POST /api/documents`, 10 MB max); searchable immediately | ✅ Implemented and tested |
| Retrieval + grounded answers with citations, sources and chunks; two "not found" checks | ✅ Implemented and tested |
| Error handling: input validation, bad files, missing index, LLM failure falls back to showing the passages | ✅ Implemented and tested |
| Web UI | ✅ Implemented |
| Structured JSON logs (latency per stage, retrieval scores, token counts) | ✅ Implemented |
| **Local mode**: runs with no AWS (on-device embeddings + local LLM via Ollama `qwen2.5:3b`) | ✅ Running |
| AWS infrastructure (CDK: S3, Lambda, Function URL, least-privilege IAM, logs) | ✅ Deployed |
| **Live AWS deployment** | ✅ Live in `us-east-2` (URL above), sample documents ingested |
| Bedrock (Titan embeddings + Claude) | 🟡 Code done, not deployed. The AWS account is on the **Free plan**, which rejects every Bedrock call ("Operation not allowed"). The deployment therefore embeds inside the Lambda and calls Groq; switching back is a config change (`EMBEDDINGS_PROVIDER=bedrock`, `LLM_PROVIDER=bedrock`) plus a re-ingest. |
| Isolation between companies | 🟡 Partly. Tenant-scoped storage paths, one index per tenant and IAM scoped to the tenant prefix are built. Authentication, per-tenant KMS keys and STS session tags are proposed ([design](docs/architecture.md#company-isolation-proposed-not-implemented)) |

## Quick start (local, no AWS)

```bash
brew install ollama && ollama serve &            # local LLM server
ollama pull qwen2.5:3b                           # ~2 GB, runs on 8 GB RAM
uv sync                                          # Python 3.12/3.13
cp .env.example .env                             # local mode by default
uv run python -m scripts.make_sample_docs        # generates data/sample/*.pdf|txt
uv run --env-file .env python -m scripts.ingest data/sample
uv run --env-file .env uvicorn app.main:app --port 8000
# open http://localhost:8000
uv run pytest                                    # 12 offline tests
```

The first ingest downloads the local embedding model (`BAAI/bge-small-en-v1.5`, about 70 MB). Running ingestion again only processes files whose content changed. On an 8 GB M-series Mac, local answers take about 1–3 s (about 12 s for the first request while the model loads). With no Ollama at all, set `LLM_PROVIDER=extractive` to show the matching passages without generated answers.

## Deploy to AWS

Requirements: an AWS profile allowed to use CloudFormation, S3, Lambda, IAM and SSM, a free [Groq](https://console.groq.com) API key, and Node.js for the CDK CLI. Docker is **not** needed.

```bash
# store the Groq key as an encrypted SecureString (copy the key first; it never touches a file or git)
aws ssm put-parameter --name /orbitus/groq-api-key --type SecureString --region us-east-2 --value "$(pbpaste)"

./scripts/build_lambda.sh                        # Linux arm64 packages + bundled embedding model (~234 MB, limit 250 MB)
npx aws-cdk@2 bootstrap                          # once per account/region
npx aws-cdk@2 deploy                             # prints AppUrl and DocsBucketName

# ingest into S3 (same embedding model as the Lambda, run from your machine)
EMBEDDINGS_PROVIDER=local STORE_URI=s3://<DocsBucketName> AWS_REGION=us-east-2 \
  uv run python -m scripts.ingest data/sample
# open AppUrl
```

The Lambda reads the key from SSM at cold start (IAM allows `ssm:GetParameter` on that one parameter only).

Tear down with `npx aws-cdk@2 destroy`. The bucket is set to delete its objects automatically.

## Example questions (sample documents)

| Question | Expected |
|---|---|
| What is the response time target for a SEV1 incident? | Acknowledge in 5 min, mitigate in 1 h (runbook p.1) |
| Who must be paged for a SEV1 incident? | Secondary on-call, Incident Commander, VP Eng (runbook p.2) |
| What caused the March 2026 database outage? | `CREATE INDEX` without `CONCURRENTLY` locked the orders table (postmortem p.1) |
| Can I deploy to production on a Friday afternoon? | No, unless it's an emergency SEV1/SEV2 fix (deployment policy p.1) |
| How often must access keys be rotated? | Every 90 days (security policy) |
| What is the parental leave policy? | **Not found**. Tests the refusal path. |
| What is the travel expense reimbursement limit? | **Not found** |

## Project layout

```
app/
  config.py      all settings come from environment variables (local vs AWS switched by config only)
  chunking.py    PDF/TXT/MD extraction + sentence-aligned chunks per page with overlap
  embeddings.py  local (fastembed) | Bedrock Titan v2
  store.py       one JSON index per tenant on local disk or S3; exact cosine search
  ingest.py      re-runnable ingestion (sha256 per file, replaces changed files)
  llm.py         Ollama (local) | Bedrock Claude (Converse) | extractive fallback; grounding prompt
  rag.py         retrieve → evidence gate → generate → NOT_FOUND gate → degrade on failure
  main.py        FastAPI routes (ask, upload), request-ID + latency middleware, Lambda handler
  static/        single-file UI
scripts/         ingest CLI, sample document generator, Lambda build
infra/           CDK stack
tests/           offline tests (fake embedder)
```

## Key design decisions

- **Lambda + Function URL instead of ECS/App Runner.** Traffic is bursty and low, so it costs close to $0 when idle and there are no servers to manage. It uses the same FastAPI code as local (via Mangum). Trade-off: cold starts of about 1–2 s, and a maximum response time of 15 min.
- **Groq on the Free plan, Bedrock as the target.** Bedrock was the first choice (IAM auth, so no secrets; data stays in AWS), but Free-plan accounts can't call it. Groq's free tier with `gpt-oss-120b` answers in about 1 s. Its key is an SSM SecureString read at cold start, so it is never in code, env files or CloudFormation. Embeddings run in the Lambda with the same ONNX model used locally, bundled in the package, so there is no download at cold start.
- **A JSON index in S3 with exact search, instead of a vector database.** For a small document set, exact cosine search in memory takes under a millisecond and has nothing to operate. Each tenant is a separate object, which gives physical isolation. The code only touches the store through `IndexStore`, so moving to S3 Vectors or OpenSearch changes one module (see Scaling).
- **Two "not found" checks.** (1) A low score floor: if nothing is even loosely related, refuse without calling the LLM (saves cost). (2) The prompt tells the model to reply `NOT_FOUND` when the passages don't support an answer, and that is the real decision. A single score threshold wasn't enough on a real uploaded NDA: "what do you know about nda" scored 0.55 (the PDF says "nondisclosure agreement", never "NDA"), while an off-topic "parental leave" question scored 0.64. With the floor at 0.5, the local model correctly refused 4 of 4 off-topic questions and answered 4 of 4 on-topic ones. The Titan floor (0.25) is only a starting point and must be re-calibrated once Bedrock access exists.
- **Chunking per page, aligned to sentences, 600 chars with 120 overlap.** Every chunk has an exact page to cite, and chunks read cleanly when shown as evidence. Trade-off: a passage that spans a page break is split.
- **Failure handling.** Bedrock clients use adaptive retries and timeouts. If the LLM fails, the user gets the relevant passages (`status: degraded`) instead of an error page. One bad file does not stop the rest of an ingestion run. The index is saved in one atomic step, and the S3 bucket is versioned.

## Discussion prep

**100 concurrent users.** Lambda scales out on its own. The bottlenecks are **Bedrock rate limits** (tokens and requests per minute) and cold starts. Fixes: request a higher Bedrock quota or provisioned throughput, use cross-region inference profiles (already on), cache answers keyed by (tenant, normalised question, index version), stream responses, and set reserved concurrency to cap spending.

**1M documents.** A single JSON index stops working: it's too big to load and exact search is too slow. Move to **S3 Vectors** (cheapest, serverless, one index per tenant) or **OpenSearch Serverless** (adds hybrid BM25+vector search and filtering). Ingestion becomes event-driven: S3 upload → SQS → Lambda or Step Functions workers with batched embedding and a dead-letter queue, plus Textract for scanned PDFs. Add hybrid retrieval and a reranker (for example Cohere Rerank on Bedrock) for precision.

**Duplicates and idempotency.** Implemented: each file is fingerprinted by sha256, unchanged files are skipped, changed files have their old chunks replaced, and chunk IDs are content hashes. At scale, add DynamoDB for per-document state, conditional writes, and SQS deduplication IDs.

**LLM fails or times out.** Implemented: 30 s read timeout, adaptive retries, then degraded mode showing the passages. Next: a circuit breaker, a fallback model ID, and streaming so users see progress.

**Secrets.** Implemented: the only secret (Groq API key) is an SSM Parameter Store SecureString (KMS-encrypted), readable only by the Lambda role, fetched once per cold start. `.env` is git-ignored. S3 access comes from the IAM role. With Bedrock there would be no secret at all. Rotation: put a new version and the next cold start picks it up.

**Isolation for 100 customers.** See the [isolation design](docs/architecture.md#company-isolation-proposed-not-implemented): the tenant comes from the auth token, one index and prefix per tenant, IAM tied to the tenant via STS session tags, a KMS key per tenant, and automated cross-tenant canary tests.

**Cost.** The main costs are LLM tokens (context passages ≫ answer), then embeddings during ingestion. Levers: the not-found check skips the LLM completely; top-k and chunk size limit prompt size; Haiku instead of larger models; ingestion that skips unchanged files avoids re-embedding; answer caching; Bedrock batch inference for bulk ingestion. Lambda and S3 cost little in comparison.

**Monitoring.** Every request logs JSON with `request_id`, `embed_ms`, `search_ms`, `llm_ms`, `total_ms`, top retrieval scores, evidence count, token counts and status (`answered`/`not_found`/`degraded`). Next: CloudWatch metric filters and alarms on the degraded rate, p95 latency, 5xx errors and Bedrock throttles; a dashboard; the not-found rate as a retrieval-quality signal; and X-Ray tracing.

## Assumptions

- The real document set wasn't supplied, so the demo uses four synthetic operations documents for a fictional company ("Northwind Cloud", `data/sample`).
- One company (tenant `demo`) is enough for the demo. Tenant handling is built in and the tenant is resolved server-side, but there is no login (the brief says no auth is needed).
- Documents are text-based PDF, TXT or Markdown in English, and small enough to index in one request.
- "Grounded" means every claim cites a retrieved passage. When the passages don't answer the question, the app says so instead of guessing.
- Region `us-east-2`. The AWS account is on the Free plan, so the deployment uses in-Lambda embeddings and Groq instead of Bedrock (see Status).

## Known limitations and shortcuts

- Bedrock isn't used in the deployment (AWS Free plan); Groq is an external service, so question text and retrieved passages leave AWS. Its free tier has rate limits; when they're hit the app shows passages (`degraded`).
- The Lambda package is close to the 250 MB limit because of the bundled ONNX runtime and model. Cold starts take a few seconds. A container image (10 GB limit) would remove the size constraint.
- No authentication (allowed by the brief). The Function URL is public, so anyone could run up Bedrock costs. Next: IAM auth or Cognito, WAF rate limiting, and reserved concurrency.
- Uploads are indexed synchronously in the request. That's fine for small files, but on Lambda a Function URL request body is limited to 6 MB. At scale: upload straight to S3 with a presigned URL → S3 event → SQS → an ingestion worker.
- There's no endpoint to delete or list documents. Remove a document by rebuilding the index.
- The whole index is loaded into memory and searched exactly, which is fine up to roughly 10⁴–10⁵ chunks.
- Scanned PDFs are rejected with a clear error; OCR is not implemented.
- Uploads in the same process are serialised with a lock, but two ingestion processes (CLI and app, or two Lambdas) writing the same tenant's index could overwrite each other (last write wins). Fix: a DynamoDB lock or a single SQS FIFO group per tenant.
- Chunks are embedded without their document title, so abbreviations that only appear in a file name (e.g. "NDA") match weakly. Adding the title to each chunk raised that score from 0.55 to 0.66 in testing; it would need a re-index.
- Scanned PDFs with a poor OCR layer produce garbled text (seen in a real NDA upload). Next: Textract OCR.
- No evaluation set beyond the calibration questions. Next: a small labelled question-and-answer set with retrieval recall@k and answer faithfulness scores.
- The sample documents are synthetic (the real document set wasn't supplied).
- Ollama is a local stand-in for Bedrock. A 3B model is weaker than Claude: with the original strict prompt it wrongly refused the Friday-deploy question, so the prompt now says "answer if a passage settles it". Sources list only the passages the answer cites.

## Time spent

About 2 hours.
