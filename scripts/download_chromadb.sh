#!/bin/bash

# ============================================================
# Script tải ChromaDB từ Google Drive về
#
# How to run:
# chmod +x scripts/download_chromadb.sh
# ./scripts/download_chromadb.sh
# ============================================================

# === CẤU HÌNH - chỉ cần sửa FILE_ID ===
GDRIVE_FILE_ID="1CVqJSbRomuNL9_v9O9IFHcB7vZjEzbtn"
ZIP_NAME="chroma_db_latest.zip"
EXTRACT_DIR="."

# === Màu terminal ===
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "=== Tải ChromaDB từ Google Drive ==="

# Kiểm tra gdown
if ! command -v gdown &> /dev/null; then
    echo "Cài gdown..."
    pip install gdown -q
fi

# Xóa zip cũ nếu có
[ -f "$ZIP_NAME" ] && rm "$ZIP_NAME"

# Tải file
echo "Đang tải..."
gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "$ZIP_NAME"

if [ $? -ne 0 ]; then
    echo -e "${RED}Tải thất bại!${NC}"
    exit 1
fi

# Xóa chroma_db cũ trước khi giải nén
echo "Đang giải nén..."
[ -d "chroma_db" ] && rm -rf chroma_db
unzip -q "$ZIP_NAME" -d "$EXTRACT_DIR"

# Xóa zip sau khi giải nén
rm "$ZIP_NAME"

echo -e "${GREEN}Hoàn thành! ChromaDB đã sẵn sàng.${NC}"