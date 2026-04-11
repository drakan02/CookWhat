import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


INPUT_PATH = "data/chunks.jsonl"
OUTPUT_DIR = "data/embeddings"
DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "bge-m3:567m"
DEFAULT_HF_MODEL = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32
REQUEST_TIMEOUT = 120
RETRY_MAX = 3
RETRY_DELAY = 2


@dataclass
class EmbedConfig:
    backend: str = DEFAULT_BACKEND
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_OLLAMA_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-10, norms)
    return matrix / safe_norms


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        end = min(start + size, len(items))
        yield start, end, items[start:end]


def _resolve_embedder(config: EmbedConfig):
    backend = config.backend.lower().strip()
    if backend == "ollama":
        return OllamaEmbedder(config)
    if backend == "hf":
        return HFEmbedder(config)
    raise ValueError(f"Backend không hợp lệ: {config.backend}. Chọn 'ollama' hoặc 'hf'.")


class OllamaEmbedder:
    def __init__(self, config: EmbedConfig):
        self.config = config

    def _url(self, endpoint: str) -> str:
        return f"{self.config.base_url}{endpoint}"

    def healthcheck(self) -> None:
        try:
            response = requests.get(self._url("/api/tags"), timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Không kết nối được Ollama tại {self.config.base_url}. Hãy chạy: ollama serve"
            ) from exc

        model_names = [m.get("name", "") for m in response.json().get("models", [])]
        accepted_names = {name.split(":")[0] for name in model_names} | set(model_names)
        if self.config.model not in accepted_names:
            raise RuntimeError(
                f"Model '{self.config.model}' chưa sẵn sàng. Các model hiện có: {model_names}"
            )

        print(f"[embedding] Ollama OK | model={self.config.model}")

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.config.model, "input": texts}
        for attempt in range(1, RETRY_MAX + 1):
            try:
                response = requests.post(
                    self._url("/api/embed"),
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                embeddings = response.json().get("embeddings")
                if not embeddings or len(embeddings) != len(texts):
                    raise ValueError(
                        f"Số embeddings trả về không khớp input: {len(embeddings) if embeddings else 0}/{len(texts)}"
                    )
                return embeddings
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt == RETRY_MAX:
                    raise RuntimeError(f"Ollama không phản hồi sau {RETRY_MAX} lần thử") from exc
                print(f"  [!] Retry {attempt}/{RETRY_MAX} sau {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(f"Ollama HTTP error: {exc}\nResponse: {response.text}") from exc

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        total_batches = (len(texts) + self.config.batch_size - 1) // self.config.batch_size
        for _, end, group in tqdm(_chunks(texts, self.config.batch_size), total=total_batches, desc="Embedding"):
            vectors.extend(self.embed_batch(group))
            if end % (self.config.batch_size * 10) == 0 or end == len(texts):
                print(f"[embedding] Encoded {end}/{len(texts)}")
        matrix = np.asarray(vectors, dtype=np.float32)
        return _l2_normalize(matrix)

    def encode_query(self, query: str) -> np.ndarray:
        vector = np.asarray(self.embed_batch([query])[0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / (norm if norm > 0 else 1e-10)


class HFEmbedder:
    def __init__(self, config: EmbedConfig):
        self.config = config
        self._model = None
        self._device = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu package cho backend 'hf'. Hãy cài: pip install sentence-transformers"
            ) from exc

        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device

        self._model = SentenceTransformer(self.config.model, device=self._device)

    def healthcheck(self) -> None:
        self._load_model()
        print(f"[embedding] HF OK | model={self.config.model} | device={self._device}")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        matrix = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(matrix, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vector, dtype=np.float32)


def _load_chunks(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if row:
                rows.append(json.loads(row))
    return rows


def _save_embedding_artifacts(output_dir: Path, chunks: list[dict], ids: list[str], embeddings: np.ndarray) -> None:
    emb_path = output_dir / "embeddings.npy"
    ids_path = output_dir / "ids.json"
    docs_path = output_dir / "documents.jsonl"

    np.save(str(emb_path), embeddings)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")

    with open(docs_path, "w", encoding="utf-8") as f:
        for item in chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[embedding] Đã lưu kết quả")
    print(f"  embeddings: {emb_path} ({embeddings.nbytes / 1e6:.1f} MB)")
    print(f"  ids       : {ids_path}")
    print(f"  documents : {docs_path}")


def embed_chunks(input_path: str, output_dir: str, config: EmbedConfig) -> None:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks: {input_path}")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    embedder = _resolve_embedder(config)
    embedder.healthcheck()

    chunks = _load_chunks(source)
    if not chunks:
        raise ValueError("Chunks rỗng, không có dữ liệu để embed")

    ids = [item["id"] for item in chunks]
    documents = [item["document"] for item in chunks]

    print(f"[embedding] Tổng chunks: {len(chunks)}")
    t0 = time.time()
    embeddings = embedder.encode_documents(documents)
    elapsed = time.time() - t0
    print(f"[embedding] Hoàn thành: shape={embeddings.shape}, time={elapsed:.1f}s")

    _save_embedding_artifacts(target_dir, chunks, ids, embeddings)

    config_path = target_dir / "embedding_config.json"
    config_path.write_text(json.dumps({
        "backend": config.backend,
        "model": config.model,
        "base_url": config.base_url,
        "device": config.device,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  config    : {config_path}")


def encode_query(
    query: str,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    device: str = "auto",
) -> np.ndarray:
    selected_model = model or (DEFAULT_HF_MODEL if backend == "hf" else DEFAULT_OLLAMA_MODEL)
    config = EmbedConfig(
        backend=backend,
        base_url=ollama_url,
        model=selected_model,
        device=device,
    )
    embedder = _resolve_embedder(config)
    return embedder.encode_query(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode chunks bằng Ollama hoặc HuggingFace")
    parser.add_argument("--input", default=INPUT_PATH, help="File chunks JSONL")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Thư mục output embeddings")
    parser.add_argument("--backend", choices=["ollama", "hf"], default=DEFAULT_BACKEND,
                        help="Backend embedding: ollama (local API) hoặc hf (GPU/CPU)")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Base URL của Ollama")
    parser.add_argument("--model", default=None,
                        help="Tên model embedding. Mặc định: bge-m3:567m (ollama) hoặc BAAI/bge-m3 (hf)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Chỉ dùng cho backend hf")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Kích thước batch")
    args = parser.parse_args()

    default_model = DEFAULT_HF_MODEL if args.backend == "hf" else DEFAULT_OLLAMA_MODEL
    cfg = EmbedConfig(
        backend=args.backend,
        base_url=args.ollama_url,
        model=args.model or default_model,
        batch_size=args.batch_size,
        device=args.device,
    )
    embed_chunks(args.input, args.output_dir, cfg)
