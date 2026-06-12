# CookWhat - Less thinking, more cooking

CookWhat là ứng dụng gợi ý món ăn bằng tiếng Việt. Người dùng nhập các nguyên liệu đang có, hệ thống tìm công thức liên quan trong cơ sở dữ liệu, dùng LLM để trả lời, và lưu lịch sử trò chuyện vào PostgreSQL.

## Tính năng

- Giao diện chat web hiện đại, tự nhiên.
- Gợi ý món ăn dựa trên nguyên liệu người dùng nhập.
- Tìm lại món theo yêu cầu mới, ví dụ món Nhật, món Hàn, món hấp, món ít calo.
- Thêm nguyên liệu vào ngữ cảnh hiện tại.
- **Text-to-Speech (TTS) offline** — đọc phản hồi bằng giọng Việt qua Piper TTS.
- Lưu lịch sử chat vào PostgreSQL.
- Tìm kiếm công thức bằng cơ chế truy xuất nhiều bước.

## Kiến trúc

```text
Browser UI
  -> FastAPI backend
      -> Intent router
      -> Ingredient extractor
      -> ChromaDB recipe search
      -> LLM response via OpenRouter
      -> PostgreSQL chat history
```

Các phần chính:

```text
main.py                 FastAPI app, API chat, API lịch sử, API TTS, static frontend
frontend/               HTML/CSS/JS giao diện chat
app/db.py               PostgreSQL storage cho chat_sessions và chat_messages
app/llm_service.py      Gọi OpenRouter LLM
app/intent_router.py    Phân loại ý định user
app/ingredient_extract.py
app/nutrition_service.py Lookup dinh dưỡng từ Vietnamese_ingredients.csv
app/prompt_builder.py   Tạo prompt trả lời gợi ý món ăn
app/utils.py            Utility functions (JSON parsing, etc.)
src/vectordb.py         ChromaDB ingest/search
src/embedding.py        Encode query/document bằng sentence-transformers
scripts/                Script tải dữ liệu, build/search vector DB
data/Vietnamese_ingredients.csv  Dữ liệu dinh dưỡng 162 thực phẩm Việt Nam
models/tts/             Model Piper TTS tiếng Việt (vi_VN-vais1000-medium)
docker-compose.yml      PostgreSQL local bằng Docker
```

## Yêu cầu

- Python 3.9+.
- Docker Desktop nếu muốn chạy PostgreSQL bằng Docker.
- OpenRouter API key.
- ChromaDB data trong thư mục `chroma_db/`, hoặc tự build lại bằng pipeline.
- **`espeak-ng`** — system dependency bắt buộc cho Piper TTS (xem hướng dẫn bên dưới).
- Ollama đang chạy với model embedding `bge-m3:567m` để encode query khi tìm kiếm.
- Vietnamese ingredients CSV trong thư mục `data/Vietnamese_ingredients.csv` cho dinh dưỡng lookup.

## Chạy ứng dụng bằng Docker (Recommended)

Bạn có thể chạy toàn bộ ứng dụng bao gồm FastAPI app, PostgreSQL database và scheduler thông qua Docker Compose mà không cần cài đặt Python hay `espeak-ng` trực tiếp trên máy của mình.

### Chuẩn bị trước khi chạy

1. Đảm bảo Docker Desktop đang chạy.
2. Đã cấu hình file `.env`.
3. Đảm bảo Ollama đang chạy trên máy host và đã tải model embedding:

   ```bash
   ollama pull bge-m3:567m
   ```

4. Tải dữ liệu ChromaDB có sẵn hoặc build dữ liệu mới (xem hướng dẫn ở mục [ChromaDB và dữ liệu công thức](#chromadb-và-dữ-liệu-công-thức) bên dưới).
5. Chuẩn bị index BM25 và các file dữ liệu (nếu chưa có):
   - **Khuyên dùng (Tải sẵn từ Google Drive)**: Chạy script sau để tải nhanh index BM25 và các file embeddings đã build sẵn:
     - Linux/macOS:

       ```bash
       chmod +x scripts/download_pipeline_files.sh
       ./scripts/download_pipeline_files.sh
       ```

     - Windows:

       ```powershell
       python scripts/download_pipeline_files.py
       ```

   - **Hoặc build thủ công** (nếu muốn tự build): Xem mục [Build BM25 Index thủ công](#build-bm25-index-thủ-công).

### Khởi động ứng dụng

Chạy lệnh sau tại thư mục gốc của dự án:

```bash
docker compose up --build -d
```

Lệnh này sẽ:

- Tự động build Docker image cho ứng dụng dựa trên `Dockerfile`.
- Khởi động 3 dịch vụ: `postgres`, `app` (FastAPI backend), và `scheduler` (hệ thống lập lịch crawl và index định kỳ).

### Truy cập ứng dụng

- **Giao diện Web (UI)**: [http://localhost:8000](http://localhost:8000)
- **Kiểm tra trạng thái (Health Check)**: [http://localhost:8000/health](http://localhost:8000/health)

## Cài đặt espeak-ng (bắt buộc cho TTS)

Piper TTS dùng `espeak-ng` để chuyển text sang phoneme. Cần cài trước khi chạy backend.

Linux (Ubuntu/Debian):

```bash
sudo apt-get install -y espeak-ng
```

macOS:

```bash
brew install espeak-ng
```

Windows:

Tải installer từ <https://github.com/espeak-ng/espeak-ng/releases> và chạy file `.msi`.

## Cài đặt Python

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn activate script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Cấu hình môi trường

Tạo file `.env` từ `.env.example`:

```powershell
Copy-Item .env.example .env
```

## PostgreSQL

Repo này dùng PostgreSQL qua Docker, không cài PostgreSQL native vào Windows.

```powershell
# Khởi động PostgreSQL
docker compose up -d postgres

# Kiểm tra container
docker ps --filter name=cookwhat-postgres

# Connection string
DATABASE_URL=postgresql://cookwhat:cookwhat_password@localhost:5432/cookwhat

# Kiểm tra bảng
docker exec cookwhat-postgres psql -U cookwhat -d cookwhat -c "\dt"

# Dừng database
docker compose stop postgres
```

## Chạy ứng dụng thủ công

Khởi động PostgreSQL trước nếu muốn lưu lịch sử:

```powershell
docker compose up -d postgres
```

Khởi động Ollama.

Chạy FastAPI:

```powershell
.\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000 --reload
# uvicorn main:app --host 127.0.0.1 --port 8000 --reload # for macos
```

Mở UI:

```text
http://127.0.0.1:8000
```

Kiểm tra backend:

```text
http://127.0.0.1:8000/health
```

Response mẫu khi PostgreSQL hoạt động:

```json
{
  "status": "running",
  "message": "CookWhat API is running",
  "postgres_history": true
}
```

Nếu `postgres_history` là `false`, app vẫn chạy nhưng lịch sử chỉ lưu trong memory và sẽ mất khi restart server.

## ChromaDB và dữ liệu công thức

Backend cần thư mục `chroma_db/` để tìm kiếm công thức. Thư mục này đang bị ignore trong Git vì có thể lớn.

### Cách 1: Tải ChromaDB có sẵn

Linux/macOS hoặc Git Bash:

```bash
chmod +x scripts/download_chromadb.sh
./scripts/download_chromadb.sh
```

Script sẽ:

- Cài `gdown` nếu thiếu.
- Tải ChromaDB từ Google Drive.
- Giải nén và thay thế thư mục `chroma_db/`.

### Cách 2: Build lại pipeline từ dữ liệu thô

Linux/macOS hoặc Git Bash:

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

Pipeline gồm:

| Bước | Module | Mục đích |
| --- | --- | --- |
| 1 | `src.chunking` | Chia dữ liệu công thức thành các đoạn nhỏ |
| 2 | `src.embedding` | Tạo embedding cho từng đoạn |
| 3 | `src.vectordb` | Nạp embedding vào ChromaDB |

Output chính:

```text
data/chunks.jsonl
data/embeddings/
chroma_db/
```

## BM25 Index và Reranker

Ứng dụng dùng **hybrid retrieval** kết hợp:

- **Dense search**: ChromaDB + BGE-M3 embeddings
- **Sparse search**: BM25 full-text search
- **Reranker**: Cross-encoder BAAI/bge-reranker-v2-m3 để xếp hạng lại kết quả
- **RRF**: Reciprocal Rank Fusion để merge kết quả từ dense + sparse

### Cài đặt dependencies

Các thư viện cần thiết cho BM25 và Reranker (`rank-bm25` và `sentence-transformers`) đã được định nghĩa sẵn trong `requirements.txt` và được cài đặt trong bước cài đặt ban đầu.

### Cách tải BM25 Index & Embeddings đã build sẵn (Khuyên dùng)

Để chatbot chạy ổn định và tránh tốn kém chi phí/tài nguyên khi chạy full pipeline embedding trên máy local, bạn có thể tải trực tiếp các file prebuilt từ Google Drive:

Linux/macOS hoặc Git Bash:

```bash
chmod +x scripts/download_pipeline_files.sh
./scripts/download_pipeline_files.sh
```

Script sẽ tự động kiểm tra và tải các file `bm25_index.pkl`, `bm25_meta.json`, `chunks.jsonl`, `documents.jsonl`, và `embeddings.npy` rồi đặt vào chính xác các thư mục tương ứng.

Mặc định, script chỉ tải khi file chưa tồn tại. Để tải lại và ghi đè các file hiện tại:

```bash
./scripts/download_pipeline_files.sh --force
```

### Build BM25 Index thủ công

> [!IMPORTANT]
> **Khuyến nghị**: Để tiết kiệm thời gian và tài nguyên máy, bạn nên tải trực tiếp BM25 index và các file pipeline đã build sẵn từ Google Drive (xem mục [Cách tải BM25 Index & Embeddings đã build sẵn (Khuyên dùng)](#cách-tải-bm25-index--embeddings-đã-build-sẵn-khuyên-dùng)). Chỉ sử dụng các bước build thủ công dưới đây nếu bạn muốn tạo lại index từ dữ liệu thô.

BM25 index được build từ `data/embeddings/documents.jsonl` (output của pipeline embedding).

**Bước 1:** Đảm bảo đã chạy pipeline embedding trước để tạo dữ liệu thô (xem mục [Cách 2: Build lại pipeline từ dữ liệu thô](#cách-2-build-lại-pipeline-từ-dữ-liệu-thô)).

**Bước 2:** Build BM25 index

Linux/macOS:

```bash
python -m scripts.build_bm25
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_bm25
```

Output sẽ được lưu tại:

```text
data/bm25/bm25_index.pkl   # BM25 index (pickle)
data/bm25/bm25_meta.json   # Mapping id → metadata + document
```

Script sẽ in ra:

```text
[bm25] Loaded 1234 documents
[bm25] Built index with tokenizer: simple_tokenize
[bm25] Saved index to data/bm25/bm25_index.pkl
[bm25] Saved metadata to data/bm25/bm25_meta.json
```

### Kiến trúc Hybrid Retrieval

Sau khi build thành công BM25, backend sẽ tự động dùng hybrid retrieval:

```
User Query
  ├─► Dense Search (ChromaDB + BGE-M3)  → top 25 candidates
  ├─► Sparse Search (BM25)              → top 25 candidates
  └─► NER Overlap Score                 → bonus signal
        ↓
   Reciprocal Rank Fusion (k=60)
        ↓
   Merged pool (~50 candidates)
        ↓
   Cross-Encoder Rerank (bge-reranker-v2-m3)
        ↓
   Final Top-5 Contexts
```

**Module chính:**

| File | Mục đích |
| --- | --- |
| `src/bm25_index.py` | Load + search BM25 index (singleton cache) |
| `src/reranker.py` | Cross-encoder reranker BAAI/bge-reranker-v2-m3 |
| `src/retriever.py` | Orchestrator: hybrid search + RRF + rerank |

## Test nhanh vector search

Windows:

```powershell
.\.venv\Scripts\python.exe -m scripts.query_vectordb
```

Linux/macOS:

```bash
.venv/bin/python -m scripts.query_vectordb
```

Script mặc định query `"gà kho gừng"` và in ra 5 kết quả gần nhất.

Bạn cũng có thể dùng trực tiếp `src.vectordb`:

```powershell
.\.venv\Scripts\python.exe -m src.vectordb search "gà kho gừng" --n 5
```

## Luồng chat

Backend phân loại message thành các intent:

```text
NEW_SEARCH       User nhập nguyên liệu mới
FOLLOW_UP        User hỏi tiếp về món đã gợi ý
RESEARCH         User muốn đổi style hoặc tìm món khác
ADD_INGREDIENT   User thêm nguyên liệu vào ngữ cảnh
SMALL_TALK       Chào hỏi, cảm ơn, tạm biệt
```

Ngữ cảnh hiện tại gồm:

```text
ingredients
recipes
```

Nếu PostgreSQL bật, ngữ cảnh và tin nhắn được lưu theo `session_id`. Nếu không, ngữ cảnh được giữ trong memory.

## Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
| --- | --- |
| `postgres_history: false` | Kiểm tra `DATABASE_URL`, chạy `docker compose up -d postgres`, restart FastAPI |
| Không mở được `localhost:8000` | Kiểm tra server uvicorn có đang chạy không |
| Port `8000` đã bị dùng | Chạy uvicorn với port khác, ví dụ `--port 8001` |
| Port `5432` đã bị dùng | Đổi mapping trong `docker-compose.yml`, ví dụ `"5433:5432"` |
| `ModuleNotFoundError` | Chạy lại `python -m pip install -r requirements.txt` trong `.venv` |
| Không có kết quả món ăn | Kiểm tra thư mục `chroma_db/` đã tồn tại và có collection `recipes` |
| Dinh dưỡng không được tìm thấy | Kiểm tra `data/Vietnamese_ingredients.csv` đã tồn tại |
| Lỗi tìm công thức / không kết nối Ollama | Chạy `ollama serve` và `ollama pull bge-m3:567m` |
| Lỗi OpenRouter/API key | Kiểm tra `OPENROUTER_API_KEY` và `OPENROUTER_MODEL` trong `.env` |
| Docker daemon chưa chạy | Mở Docker Desktop rồi chạy lại lệnh Docker |
| Nút đọc không có tiếng | Kiểm tra `espeak-ng` đã cài chưa; xem log uvicorn có `[TTS] Piper model loaded` không |
| `TTS model chưa được load` (503) | File model thiếu trong `models/tts/`; chạy lại `git pull` để lấy file model |
| `FileNotFoundError: Không tìm thấy BM25 index` | Tải sẵn qua `download_pipeline_files.sh`/`.py` (Khuyên dùng) hoặc chạy `python -m scripts.build_bm25` |
| `FileNotFoundError: Không tìm thấy documents.jsonl` | Tải sẵn qua `download_pipeline_files.sh`/`.py` (Khuyên dùng) hoặc chạy `./scripts/run_pipeline.sh` |
| Reranker tải lâu lần đầu | Bình thường, download model Hugging Face từ internet. Lần sau nhanh hơn |
