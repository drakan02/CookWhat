import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Thêm root path vào sys.path để import app và src khi chạy script trực tiếp
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Cấu hình LOGGING
LOG_DIR = BASE_DIR / "cookpad_data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("periodic_pipeline")

try:
    from app import db
    from src.chunking import recipe_to_text, build_metadata
    from src.embedding import encode_documents, check_ollama_connection
    from src.vectordb import get_collection, BATCH_SIZE
    from scripts.build_bm25 import build_bm25_index
except ImportError as e:
    logger.error(f"Không thể import các module hệ thống. Hãy chạy script từ thư mục root của dự án. Lỗi: {e}")
    sys.exit(1)


def run_crawler(test_mode: bool = False) -> Path:
    """Gọi script crawl để lấy dữ liệu các công thức mới nhất."""
    logger.info("=== BẮT ĐẦU CRAWL TỪ COOKPAD.COM ===")
    
    crawl_script = BASE_DIR / "scripts" / "crawl_cookpad.py"
    if not crawl_script.exists():
        raise FileNotFoundError(f"Không tìm thấy crawler script tại: {crawl_script}")
    
    cmd = [
        sys.executable,
        str(crawl_script),
        "--mode", "recent",
        "--output-dir", str(LOG_DIR)
    ]
    
    if test_mode:
        # Trong chế độ test, giới hạn số trang quét và số worker để chạy nhanh
        cmd.extend(["--start-page", "1", "--end-page", "1", "--workers", "4"])
        logger.info("Chạy crawler trong chế độ TEST_MODE (chỉ quét trang 1)...")
    
    # Thực thi tiến trình cào
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode != 0:
        logger.error("Crawler gặp lỗi khi thực thi:")
        logger.error(result.stderr)
        raise RuntimeError(f"Crawl failed with return code {result.returncode}")
        
    logger.info("Crawler hoàn thành thành công.")
    return LOG_DIR / "recipes.jsonl"


def sync_recipes_to_db(recipes_file: Path) -> int:
    """Đọc dữ liệu từ file cào được và cập nhật vào PostgreSQL raw_recipes."""
    logger.info("=== ĐỒNG BỘ DỮ LIỆU CÀO ĐƯỢC VÀO POSTGRESQL ===")
    
    if not recipes_file.exists():
        logger.warning(f"Không tìm thấy file kết quả cào: {recipes_file}")
        return 0

    # Khởi tạo DB nếu chưa
    db.init_db()
    if not db.ready():
        raise RuntimeError("PostgreSQL chưa sẵn sàng. Không thể đồng bộ dữ liệu.")

    count = 0
    with open(recipes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recipe = json.loads(line)
                recipe_id = str(recipe.get("id"))
                url = recipe.get("url", "")
                title = recipe.get("title", "")
                
                if not recipe_id or not title:
                    continue
                
                # Upsert vào bảng raw_recipes
                db.upsert_raw_recipe(recipe_id, url, title, recipe)
                count += 1
            except Exception as e:
                logger.error(f"Lỗi khi lưu recipe vào DB: {e}")
                
    logger.info(f"Đã đồng bộ {count} công thức vào PostgreSQL raw_recipes.")
    return count


def process_indexing() -> None:
    """Xử lý chunking, sinh embedding và lập chỉ mục cho các công thức chưa index."""
    logger.info("=== LẬP CHỈ MỤC TĂNG TRƯỞNG (INCREMENTAL INDEXING) ===")
    
    db.init_db()
    if not db.ready():
        raise RuntimeError("PostgreSQL chưa sẵn sàng.")

    # 1. Lấy danh sách công thức chưa index
    unindexed = db.get_unindexed_recipes()
    if not unindexed:
        logger.info("Không có công thức mới nào cần lập chỉ mục.")
        return

    logger.info(f"Phát hiện {len(unindexed)} công thức mới cần lập chỉ mục.")

    # 2. Sinh chunk từ các công thức mới
    new_chunks = []
    for recipe in unindexed:
        new_chunks.append({
            "id": str(recipe["id"]),
            "document": recipe_to_text(recipe),
            "metadata": build_metadata(recipe)
        })

    # 3. Tạo vector embeddings qua Ollama
    logger.info("Đang gọi Ollama sinh embeddings...")
    check_ollama_connection()
    texts = [c["document"] for c in new_chunks]
    new_embeddings = encode_documents(texts)  # (M, dim)
    
    # 4. Upsert vào ChromaDB
    logger.info("Đang nạp vector mới vào ChromaDB...")
    collection = get_collection()
    total = len(new_chunks)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_ids = [c["id"] for c in new_chunks[start:end]]
        batch_embs = new_embeddings[start:end].tolist()
        batch_docs = [c["document"] for c in new_chunks[start:end]]
        batch_metas = [c["metadata"] for c in new_chunks[start:end]]
        
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_metas
        )

    # 5. Cập nhật các file backup trên disk (embeddings.npy & ids.json)
    logger.info("Cập nhật file lưu trữ cục bộ (embeddings.npy & ids.json)...")
    emb_dir = BASE_DIR / "data" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    emb_path = emb_dir / "embeddings.npy"
    ids_path = emb_dir / "ids.json"
    
    if emb_path.exists() and ids_path.exists():
        try:
            old_embeddings = np.load(str(emb_path))
            with open(ids_path, encoding="utf-8") as f:
                old_ids = json.load(f)
            
            # Loại bỏ trùng lặp nếu ID mới đã có trong file cũ
            new_ids_set = set(c["id"] for c in new_chunks)
            keep_indices = [i for i, oid in enumerate(old_ids) if oid not in new_ids_set]
            
            filtered_embeddings = old_embeddings[keep_indices]
            filtered_ids = [old_ids[i] for i in keep_indices]
            
            # Ghép nối vector mới
            updated_embeddings = np.concatenate([filtered_embeddings, new_embeddings], axis=0)
            updated_ids = filtered_ids + [c["id"] for c in new_chunks]
        except Exception as e:
            logger.warning(f"Lỗi khi đọc file backup, tạo mới file backup: {e}")
            updated_embeddings = new_embeddings
            updated_ids = [c["id"] for c in new_chunks]
    else:
        updated_embeddings = new_embeddings
        updated_ids = [c["id"] for c in new_chunks]

    np.save(str(emb_path), updated_embeddings)
    ids_path.write_text(json.dumps(updated_ids, ensure_ascii=False), encoding="utf-8")

    # 6. Đánh dấu đã lập chỉ mục trong PostgreSQL
    new_ids = [c["id"] for c in new_chunks]
    db.mark_recipes_as_indexed(new_ids)
    logger.info(f"Đã cập nhật trạng thái index cho {len(new_ids)} công thức trong PostgreSQL.")

    # 7. Tải toàn bộ indexed recipes để ghi đè file master documents.jsonl và rebuild BM25
    logger.info("=== KHỞI TẠO LẠI CHỈ MỤC TÌM KIẾM BM25 ===")
    all_indexed = db.get_all_indexed_recipes()
    logger.info(f"Tổng số công thức đã được lập chỉ mục: {len(all_indexed)}")
    
    all_chunks = []
    for r in all_indexed:
        all_chunks.append({
            "id": str(r["id"]),
            "document": recipe_to_text(r),
            "metadata": build_metadata(r)
        })
        
    doc_path = emb_dir / "documents.jsonl"
    with open(doc_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    # Gọi hàm rebuild BM25 index
    build_bm25_index(documents_path=str(doc_path))
    logger.info("BM25 Index được xây dựng thành công.")


def main():
    parser = argparse.ArgumentParser(description="Periodic Crawler & Indexer Pipeline Orchestrator.")
    parser.add_argument("--test-mode", action="store_true", help="Chạy pipeline trong chế độ TEST (quét ít dữ liệu hơn)")
    args = parser.parse_args()
    
    t0 = time.time()
    logger.info("==================================================================")
    logger.info(f"BẮT ĐẦU CHẠY PIPELINE ĐỊNH KỲ (Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info("==================================================================")
    
    try:
        # Bước 1: Crawl
        recipes_file = run_crawler(test_mode=args.test_mode)
        
        # Bước 2: Đồng bộ DB Staging
        sync_recipes_to_db(recipes_file)
        
        # Bước 3: Ingest ChromaDB & BM25 Index
        process_indexing()
        
        elapsed = time.time() - t0
        logger.info("==================================================================")
        logger.info(f"PIPELINE HOÀN THÀNH THÀNH CÔNG! Tổng thời gian chạy: {elapsed:.1f}s")
        logger.info("==================================================================")
    except Exception as e:
        logger.exception(f"PIPELINE GẶP LỖI NGHIÊM TRỌNG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()
