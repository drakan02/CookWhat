#!/usr/bin/env python3
"""
Script tải các file dữ liệu (BM25 index, embeddings, chunks) từ Google Drive
và tự động đặt vào các thư mục tương ứng trong dự án.
Chỉ tải khi file chưa tồn tại (hoặc khi dùng cờ --force).

Cách chạy:
    python -m scripts.download_pipeline_files
Hoặc:
    python scripts/download_pipeline_files.py
"""

import os
import argparse
import logging
from pathlib import Path
import gdown

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# MAPPING CẤU HÌNH FILE & GOOGLE DRIVE ID
# ──────────────────────────────────────────────
# Map đường dẫn đích trong project với Google Drive File ID tương ứng
FILES_TO_DOWNLOAD = {
    "data/bm25/bm25_index.pkl": "1fhn9ninAZ3whUGWZqvaWGp39FYx_EH0P",
    "data/bm25/bm25_meta.json": "1-LpimQHzP9zbfn_Hzt4aYrdRttwezEZo",
    "data/chunks.jsonl": "1jFAzkV7way_LQe3kYe3IqXXS6vOo-guG",
    "data/embeddings/documents.jsonl": "1vQ9cWOsx9zhjrEHlTFAVP_jbjqSLN-T2",
    "data/embeddings/embeddings.npy": "1eJZ4cky9UHtdQMyaiV8w75yrdUxLzkuF",
}


def download_files(force_download: bool = False):
    """Tải các file dữ liệu từ Google Drive nếu chưa tồn tại."""
    logger.info("=== Bắt đầu kiểm tra và tải dữ liệu từ Google Drive ===")
    
    download_count = 0
    skip_count = 0

    for relative_path, file_id in FILES_TO_DOWNLOAD.items():
        dest_path = Path(relative_path)
        google_drive_url = f"https://drive.google.com/uc?id={file_id}"

        # Kiểm tra sự tồn tại của file
        if dest_path.exists() and not force_download:
            logger.info(f"[ĐÃ TỒN TẠI] {dest_path} -> Bỏ qua.")
            skip_count += 1
            continue

        # Đảm bảo thư mục cha tồn tại
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ĐANG TẢI] {dest_path}...")
        logger.info(f"  ID: {file_id}")
        logger.info(f"  Lưu tại: {dest_path}")

        try:
            # Tải file sử dụng gdown
            # quiet=False để hiển thị thanh tiến trình tải trực quan
            gdown.download(google_drive_url, str(dest_path), quiet=False)
            
            if dest_path.exists():
                logger.info(f"[THÀNH CÔNG] Đã lưu: {dest_path} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                download_count += 1
            else:
                logger.error(f"[LỖI] Tải thành công nhưng không thấy file ở: {dest_path}")
        except Exception as e:
            logger.error(f"[LỖI] Không thể tải {dest_path}: {e}")
            raise

    logger.info("==================================================")
    logger.info(f"Hoàn thành! Đã tải: {download_count} file, bỏ qua: {skip_count} file.")
    logger.info("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tải các file pipeline BM25 & Embeddings từ Google Drive."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc tải lại và ghi đè các file đã tồn tại.",
    )
    args = parser.parse_args()

    download_files(force_download=args.force)
