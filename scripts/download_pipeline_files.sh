#!/bin/bash

# ============================================================
# Script wrapper để chạy download_pipeline_files.py
#
# How to run:
# chmod +x scripts/download_pipeline_files.sh
# ./scripts/download_pipeline_files.sh
# ============================================================

# Chạy python script
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/download_pipeline_files.py "$@"
else
    python3 scripts/download_pipeline_files.py "$@"
fi
