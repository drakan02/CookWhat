#!/bin/bash
# Run from chunking to chromaDB
echo "=== Bước 1: Chunking ==="
.venv/bin/python -m src.chunking

echo "=== Bước 2: Embedding ==="
.venv/bin/python -m src.embedding

echo "=== Bước 3: Nạp vào ChromaDB ==="
.venv/bin/python -m src.vectordb ingest --embeddings-dir data/embeddings --reset

echo "=== Xong! ==="

# How to run:
# chmod +x scripts/run_pipeline.sh
# ./scripts/run_pipeline.sh

# Zip ChromaDB file (according to time):
# zip -r chroma_db_$(date +%Y%m%d).zip chroma_db/