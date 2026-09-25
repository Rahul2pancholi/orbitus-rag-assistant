"""Ingest a directory (or files) into the configured tenant's index.

    uv run --env-file .env python -m scripts.ingest data/sample
    uv run --env-file .env python -m scripts.ingest data/sample --tenant acme
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import load_settings
from app.ingest import ingest_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--tenant", help="tenant id (defaults to TENANT_ID)")
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if not p.exists():
            parser.error(f"{p} does not exist")
        files.extend(sorted(p.iterdir()) if p.is_dir() else [p])

    report = ingest_paths(files, load_settings(), tenant_id=args.tenant)
    print(json.dumps(report, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
