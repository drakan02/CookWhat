import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from step2_embedding import encode_query


EMBEDDINGS_DIR = "data/embeddings"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "vietnamese_recipes"
INGEST_BATCH = 512
DEFAULT_EMBED_BACKEND = "ollama"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "bge-m3:567m"
DEFAULT_HF_MODEL = "BAAI/bge-m3"


@dataclass
class StoreConfig:
    chroma_path: str = CHROMA_PATH
    collection_name: str = COLLECTION_NAME
    embed_backend: str = DEFAULT_EMBED_BACKEND
    embed_model: str = DEFAULT_OLLAMA_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    device: str = "auto"


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _contains_icase(text: str, keyword: str | None) -> bool:
    if not keyword:
        return True
    return keyword.casefold() in str(text).casefold()


class RecipeVectorStore:
    def __init__(self, config: StoreConfig):
        self.config = config
        self.client = chromadb.PersistentClient(
            path=self.config.chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection_for_ingest(self, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(self.config.collection_name)
                print(f"[vectordb] Đã reset collection: {self.config.collection_name}")
            except Exception:
                pass
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _collection_for_search(self):
        return self.client.get_collection(self.config.collection_name)

    def ingest(self, embeddings_dir: str, reset: bool = False) -> None:
        base_dir = Path(embeddings_dir)
        emb_path = base_dir / "embeddings.npy"
        ids_path = base_dir / "ids.json"
        docs_path = base_dir / "documents.jsonl"

        for file_path in [emb_path, ids_path, docs_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Thiếu file đầu vào: {file_path}")

        vectors = np.load(str(emb_path))
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        chunks = _read_jsonl(docs_path)

        if not (len(vectors) == len(ids) == len(chunks)):
            raise ValueError(
                f"Số lượng không khớp embeddings={len(vectors)}, ids={len(ids)}, chunks={len(chunks)}"
            )

        collection = self._collection_for_ingest(reset=reset)
        total = len(ids)
        print(f"[vectordb] Bắt đầu ingest {total} documents")

        for start in range(0, total, INGEST_BATCH):
            end = min(start + INGEST_BATCH, total)
            collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=[item["document"] for item in chunks[start:end]],
                metadatas=[item["metadata"] for item in chunks[start:end]],
            )
            print(f"[vectordb] Ingested {end}/{total}")

        print(f"[vectordb] Hoàn thành, tổng documents: {collection.count()}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_ingredient: str | None = None,
        filter_location: str | None = None,
    ) -> list[dict]:
        collection = self._collection_for_search()
        query_vector = encode_query(
            query,
            backend=self.config.embed_backend,
            model=self.config.embed_model,
            ollama_url=self.config.ollama_url,
            device=self.config.device,
        ).tolist()

        # Lấy dư ứng viên để filter mềm phía Python, tránh phụ thuộc operator của Chroma.
        raw_limit = max(n_results * 5, n_results)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=raw_limit,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if not _contains_icase(meta.get("ner", ""), filter_ingredient):
                continue
            if not _contains_icase(meta.get("author_location", ""), filter_location):
                continue

            output.append(
                {
                    "rank": len(output) + 1,
                    "id": doc_id,
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "cook_time": meta.get("cook_time", ""),
                    "ner": meta.get("ner", ""),
                    "score": round(1 - dist, 4),
                    "document": doc,
                    "metadata": meta,
                }
            )
            if len(output) >= n_results:
                break
        return output


def ingest(
    embeddings_dir: str = EMBEDDINGS_DIR,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> None:
    store = RecipeVectorStore(StoreConfig(chroma_path=chroma_path, collection_name=collection_name))
    store.ingest(embeddings_dir=embeddings_dir, reset=reset)


def search(
    query: str,
    n_results: int = 5,
    filter_ingredient: str | None = None,
    filter_location: str | None = None,
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    device: str = "auto",
) -> list[dict]:
    store = RecipeVectorStore(
        StoreConfig(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embed_backend=embed_backend,
            embed_model=embed_model,
            ollama_url=ollama_url,
            device=device,
        )
    )
    return store.search(
        query=query,
        n_results=n_results,
        filter_ingredient=filter_ingredient,
        filter_location=filter_location,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest và search recipe vectors bằng ChromaDB")
    subparsers = parser.add_subparsers(dest="command")

    p_ingest = subparsers.add_parser("ingest", help="Nạp embeddings vào ChromaDB")
    p_ingest.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    p_ingest.add_argument("--chroma-path", default=CHROMA_PATH)
    p_ingest.add_argument("--collection-name", default=COLLECTION_NAME)
    p_ingest.add_argument("--reset", action="store_true")

    p_search = subparsers.add_parser("search", help="Tìm kiếm recipe")
    p_search.add_argument("query")
    p_search.add_argument("--n", type=int, default=5)
    p_search.add_argument("--ingredient", default=None)
    p_search.add_argument("--location", default=None)
    p_search.add_argument("--chroma-path", default=CHROMA_PATH)
    p_search.add_argument("--collection-name", default=COLLECTION_NAME)
    p_search.add_argument("--backend", choices=["ollama", "hf"], default=DEFAULT_EMBED_BACKEND)
    p_search.add_argument("--model", default=None,
                          help="Mặc định: bge-m3:567m (ollama) hoặc BAAI/bge-m3 (hf)")
    p_search.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p_search.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(
            embeddings_dir=args.embeddings_dir,
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            reset=args.reset,
        )
    elif args.command == "search":
        selected_model = args.model or (DEFAULT_HF_MODEL if args.backend == "hf" else DEFAULT_OLLAMA_MODEL)
        rows = search(
            query=args.query,
            n_results=args.n,
            filter_ingredient=args.ingredient,
            filter_location=args.location,
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            embed_backend=args.backend,
            embed_model=selected_model,
            ollama_url=args.ollama_url,
            device=args.device,
        )
        print(f"\nCâu hỏi: {args.query}")
        print(f"Top {len(rows)} kết quả:\n")
        for row in rows:
            print(f"{'-' * 60}")
            print(f"#{row['rank']} [{row['score']:.4f}] {row['title']}")
            print(f"  URL: {row['url']}")
            print(f"  Nấu: {row['cook_time']}")
            print(f"  NER: {row['ner']}")
    else:
        parser.print_help()