# Smart Recipe Recommendation System (SRRS)

Pipeline 3 bước cho bài toán gợi ý công thức nấu ăn bằng semantic search:

1. Chunking dữ liệu recipe thành văn bản chuẩn để embed
2. Tạo embedding bằng Ollama (`bge-m3:567m`)
3. Nạp vector vào ChromaDB và truy vấn tìm kiếm

## Cấu trúc dự án

- `step1_chunking.py`: Đọc JSONL/JSON recipes và xuất `data/chunks.jsonl`
- `step2_embedding.py`: Đọc chunks, gọi Ollama để embed, lưu vào `data/embeddings/`
- `step3_vector_database.py`: Ingest vào ChromaDB và search

## Yêu cầu

- Python 3.10+
- Ollama đã cài đặt

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull bge-m3:567m
ollama serve
```

## Chạy pipeline

### Bước 1: Chunking

```bash
python step1_chunking.py --input data/recipes.jsonl --output data/chunks.jsonl
```

### Bước 2: Embedding

```bash
python step2_embedding.py --input data/chunks.jsonl --output-dir data/embeddings
```

### Bước 3: Ingest vào ChromaDB

```bash
python step3_vector_database.py ingest --embeddings-dir data/embeddings --reset
```

### Search

```bash
python step3_vector_database.py search "món ăn nhẹ từ cá" --n 5
python step3_vector_database.py search "phở bò" --n 3 --ingredient "gầu bò"
python step3_vector_database.py search "món miền Nam" --location "Hồ Chí Minh"
```
