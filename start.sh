#!/bin/bash
set -e

echo "==> Checking ChromaDB collection..."

python - <<'EOF'
import os, sys
sys.path.insert(0, '/app')
from backend.config import CHROMA_DIR, COLLECTION_NAME
import chromadb

os.makedirs(CHROMA_DIR, exist_ok=True)

try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION_NAME)
    count = col.count()
    if count > 0:
        print(f"Collection '{COLLECTION_NAME}' has {count} chunks — skipping ingest.")
        sys.exit(0)
    else:
        print(f"Collection '{COLLECTION_NAME}' is empty — will run ingest.")
        sys.exit(1)
except Exception as e:
    print(f"Collection not found ({e}) — will run ingest.")
    sys.exit(1)
EOF

CHECK_EXIT=$?

if [ $CHECK_EXIT -ne 0 ]; then
    echo "==> Running ingest..."
    python backend/ingest.py
fi

echo "==> Starting server on port ${PORT:-8000}..."
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"
