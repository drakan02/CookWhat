# CookWhat


## Yêu Cầu Cài Đặt

Đảm bảo đã cài **Python 3.9+** và thiết lập môi trường ảo.

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## Scripts

### 1. `download_chromadb.sh` — Tải ChromaDB có sẵn

Tải xuống bản ChromaDB đã được xây dựng sẵn từ Google Drive, giúp truy vấn ngay lập tức.

**Chạy:**
```bash
chmod +x scripts/download_chromadb.sh
./scripts/download_chromadb.sh
```

**Script này sẽ:**
- Tự động cài `gdown` nếu chưa có
- Tải file ZIP `chroma_db` mới nhất từ Google Drive
- Thay thế thư mục `chroma_db/` cũ bằng dữ liệu vừa tải về

> **Lưu ý:** Cần có kết nối Internet và đủ dung lượng ổ đĩa. Dùng script này nếu muốn bỏ qua bước 2 và truy vấn ngay.

---

### 2. `run_pipeline.sh` — Chạy Toàn Bộ Pipeline Dữ Liệu

Chạy pipeline 3 bước để xây dựng cơ sở dữ liệu vector ChromaDB từ đầu:

| Bước | Module | Mô tả |
|------|--------|--------|
| 1 | `src.chunking` | Chia dữ liệu công thức thành các đoạn văn bản nhỏ |
| 2 | `src.embedding` | Tạo embedding câu cho từng đoạn |
| 3 | `src.vectordb` | Nạp toàn bộ embedding vào ChromaDB |

**Chạy:**
```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

**Script này sẽ:**
- Đọc dữ liệu thô từ `data/`
- Ghi embedding vào `data/embeddings/`
- Xóa và xây dựng lại cơ sở dữ liệu vector `chroma_db/`

> **Lưu ý:** Quá trình này có thể mất vài phút tùy thuộc vào kích thước dữ liệu và phần cứng. Chạy script này khi có dữ liệu công thức mới hoặc đã được cập nhật.

**Tùy chọn — Xuất ChromaDB thành file ZIP**:
```bash
zip -r chroma_db_$(date +%Y%m%d).zip chroma_db/
```

---

### 3. `query_vectordb.py` — Truy Vấn Cơ Sở Dữ Liệu Vector

Script mẫu thực hiện tìm kiếm ngữ nghĩa trên ChromaDB và in ra các kết quả phù hợp nhất.

**Chạy (từ thư mục gốc của dự án):**
```bash
.venv/bin/python -m scripts.query_vectordb
```

**Truy vấn mặc định:** `"gà kho gừng"` — trả về 5 công thức có ngữ nghĩa tương đồng nhất.

**Ví dụ kết quả:**
```
──────────────────────────────────────────────────
#1 Score: 0.9234
    Title: Gà Kho Gừng Đậm Đà
    URL  : https://...
    NER  : gà, gừng, nước mắm
    Doc  : Gà kho gừng là món ăn đậm đà, thơm ngon...
```

> **Lưu ý:** ChromaDB phải sẵn sàng trước khi chạy script này. Hãy chạy `run_pipeline.sh` (bước 2) hoặc `download_chromadb.sh` (bước 1) trước.

Để thay đổi câu truy vấn hoặc bộ lọc, chỉnh sửa `scripts/query_vectordb.py`:

```python
results = search(
    query="câu truy vấn của bạn",   # Thay đổi tại đây
    n_results=5,                    # Số lượng kết quả
    filter_ingredient=None,         # Ví dụ: "gà" để lọc theo nguyên liệu
    filter_location=None,           # Ví dụ: "Hà Nội" để lọc theo vùng miền
)
```

---

## Xử Lý Sự Cố

| Vấn đề | Giải pháp |
|--------|-----------|
| Không tìm thấy `gdown` | Chạy `pip install gdown` hoặc để `download_chromadb.sh` tự cài |
| `Permission denied` với file `.sh` | Chạy `chmod +x scripts/*.sh` |
| `ModuleNotFoundError` | Đảm bảo bạn đang dùng `.venv/bin/python`, không phải Python hệ thống |
| ChromaDB trống hoặc không tồn tại | Chạy `download_chromadb.sh` hoặc `run_pipeline.sh` trước |