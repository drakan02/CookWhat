import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings


EMBEDDINGS_DIR = "/kaggle/working/data/embeddings"
CHROMA_PATH = "/kaggle/working/chroma_db"
COLLECTION_NAME = "vietnamese_recipes"
INGEST_BATCH = 512
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DEVICE = "auto"

_ENCODER_CACHE: dict[tuple[str, str], object] = {}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_query(query: str, model: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE) -> np.ndarray:
    key = (model, _resolve_device(device))
    encoder = _ENCODER_CACHE.get(key)

    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Thieu package sentence-transformers. Hay cai: pip install sentence-transformers"
            ) from exc

        encoder = SentenceTransformer(model, device=key[1])
        _ENCODER_CACHE[key] = encoder

    vector = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vector, dtype=np.float32)


@dataclass
class StoreConfig:
    chroma_path: str = CHROMA_PATH
    collection_name: str = COLLECTION_NAME
    model: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE


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
                print(f"[vectordb] Reset collection: {self.config.collection_name}")
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
                raise FileNotFoundError(f"Thieu file dau vao: {file_path}")

        vectors = np.load(str(emb_path))
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        chunks = _read_jsonl(docs_path)

        if not (len(vectors) == len(ids) == len(chunks)):
            raise ValueError(
                f"So luong khong khop embeddings={len(vectors)}, ids={len(ids)}, chunks={len(chunks)}"
            )

        collection = self._collection_for_ingest(reset=reset)
        total = len(ids)
        print(f"[vectordb] Ingest {total} documents")

        for start in range(0, total, INGEST_BATCH):
            end = min(start + INGEST_BATCH, total)
            collection.upsert(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=[item["document"] for item in chunks[start:end]],
                metadatas=[item["metadata"] for item in chunks[start:end]],
            )
            print(f"[vectordb] Ingested {end}/{total}")

        print(f"[vectordb] Done. total docs: {collection.count()}")

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
            model=self.config.model,
            device=self.config.device,
        ).tolist()

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
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> list[dict]:
    store = RecipeVectorStore(
        StoreConfig(
            chroma_path=chroma_path,
            collection_name=collection_name,
            model=model,
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
    parser = argparse.ArgumentParser(description="Kaggle Chroma ingest/search")
    subparsers = parser.add_subparsers(dest="command")

    p_ingest = subparsers.add_parser("ingest", help="Nap embeddings vao ChromaDB")
    p_ingest.add_argument("--embeddings-dir", default=EMBEDDINGS_DIR)
    p_ingest.add_argument("--chroma-path", default=CHROMA_PATH)
    p_ingest.add_argument("--collection-name", default=COLLECTION_NAME)
    p_ingest.add_argument("--reset", action="store_true")

    p_search = subparsers.add_parser("search", help="Tim kiem recipe")
    p_search.add_argument("query")
    p_search.add_argument("--n", type=int, default=5)
    p_search.add_argument("--ingredient", default=None)
    p_search.add_argument("--location", default=None)
    p_search.add_argument("--chroma-path", default=CHROMA_PATH)
    p_search.add_argument("--collection-name", default=COLLECTION_NAME)
    p_search.add_argument("--model", default=DEFAULT_MODEL)
    p_search.add_argument("--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE)

    # Strip kernel-injected args such as: -f /tmp/<id>.json and HistoryManager flags.
    raw_args = sys.argv[1:]
    cleaned_args: list[str] = []
    skip_next = False
    for idx, token in enumerate(raw_args):
        if skip_next:
            skip_next = False
            continue
        if token == "-f":
            skip_next = True
            continue
        if token.startswith("--HistoryManager"):
            continue
        if idx > 0 and raw_args[idx - 1] == "-f":
            continue
        cleaned_args.append(token)

    # Avoid argparse SystemExit in notebook cells when no subcommand is passed.
    if not cleaned_args or cleaned_args[0] not in {"ingest", "search"}:
        print("[vectordb] No CLI command provided. Use 'ingest' or 'search' when running as script.")
        print("[vectordb] Example: !python kaggle_step3_vector_database.py ingest --reset")
        raise SystemExit(0)

    args, _ = parser.parse_known_args(cleaned_args)

    if args.command == "ingest":
        ingest(
            embeddings_dir=args.embeddings_dir,
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            reset=args.reset,
        )
    elif args.command == "search":
        rows = search(
            query=args.query,
            n_results=args.n,
            filter_ingredient=args.ingredient,
            filter_location=args.location,
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            model=args.model,
            device=args.device,
        )
        print(f"\nQuery: {args.query}")
        print(f"Top {len(rows)} results:\n")
        for row in rows:
            print(f"{'-' * 60}")
            print(f"#{row['rank']} [{row['score']:.4f}] {row['title']}")
            print(f"  URL: {row['url']}")
            print(f"  Time: {row['cook_time']}")
            print(f"  NER: {row['ner']}")
    else:
        parser.print_help()
