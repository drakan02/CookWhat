import gdown
import os
import logging
from pathlib import Path

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
# CONFIG
# ──────────────────────────────────────────────

# Google Drive file ID
GOOGLE_DRIVE_FILE_ID = "1ElAHB4hMmPcaYCOe2Se72JSi-M5BeiyS"
GOOGLE_DRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

# Thư mục lưu dữ liệu
DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "data.jsonl" 


# ──────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────

def download_data():
    """Tải dữ liệu từ Google Drive."""
    
    # Tạo thư mục nếu chưa tồn tại
    DATA_DIR.mkdir(exist_ok=True)
    
    # Kiểm tra nếu file đã tồn tại
    if OUTPUT_FILE.exists():
        logger.info(f"File đã tồn tại: {OUTPUT_FILE}")
        return
    
    logger.info(f"Đang tải dữ liệu từ Google Drive...")
    logger.info(f"URL: {GOOGLE_DRIVE_URL}")
    logger.info(f"Lưu vào: {OUTPUT_FILE}")
    
    try:
        gdown.download(GOOGLE_DRIVE_URL, str(OUTPUT_FILE), quiet=False)
        logger.info(f"Tải thành công! Dữ liệu lưu tại: {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        raise


if __name__ == "__main__":
    download_data()
