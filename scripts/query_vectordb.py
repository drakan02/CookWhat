from src.vectordb import search

results = search(
    query="gà kho gừng",
    n_results=5,
    filter_ingredient=None,
    filter_location=None,
)

for i, r in enumerate(results, 1):
    print(f"\n{'─'*50}")
    print(f"#{i} Score: {r['score']:.4f}")
    print(f"    Title: {r['title']}")
    print(f"    URL  : {r['url']}")
    print(f"    NER  : {r['metadata'].get('ner', '')}")
    print(f"    Doc  : {r['document'][:200]}...")


# How to run:
# From root folder:.venv/bin/python -m scripts.query_vectordb
# .venv/bin/python -m scripts.query_vectordb