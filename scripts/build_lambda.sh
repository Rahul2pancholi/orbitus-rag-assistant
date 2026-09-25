#!/usr/bin/env bash
# Build the Lambda deployment package into build/lambda without Docker:
# uv resolves Linux/arm64 wheels directly. Only runtime deps are shipped (no uvicorn).
# The embedding model is bundled so the Lambda never downloads it at cold start.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf build/lambda
mkdir -p build/lambda
uv export --no-default-groups --no-hashes --no-emit-project --format requirements-txt > build/requirements.txt
uv pip install \
  --requirements build/requirements.txt \
  --target build/lambda \
  --python-platform aarch64-manylinux_2_28 \
  --python-version 3.13 \
  --only-binary :all: \
  --quiet
cp -R app build/lambda/app
uv run python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='build/model-cache')"
cp -RL build/model-cache/models--*/snapshots/*/ build/lambda/embed-model  # real files, no symlinks
# boto3 is already in the Lambda runtime; hf_xet is only used for downloads. Both push the package over 250 MB.
rm -rf build/lambda/{boto3,botocore,s3transfer,hf_xet}
find build/lambda -name "__pycache__" -type d -prune -exec rm -rf {} +
echo "built build/lambda ($(du -sh build/lambda | cut -f1))"
