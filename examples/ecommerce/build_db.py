"""
build_db.py  —  index ecommerce products into a local ChromaDB collection.

Run once before searching:
    python build_db.py
"""

import chromadb
from chromadb.utils import embedding_functions
from products import PRODUCTS

COLLECTION_NAME = "ecommerce_products"
DB_PATH = "./chroma_db"


def build():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Remove old collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # Default embedding function uses all-MiniLM-L6-v2 (runs locally, no API key)
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [p["id"] for p in PRODUCTS]
    documents = [p["description"] for p in PRODUCTS]

    # ChromaDB metadata values must be str / int / float / bool — flatten lists
    metadatas = []
    for p in PRODUCTS:
        flat = {}
        for k, v in p["metadata"].items():
            if isinstance(v, list):
                flat[k] = ",".join(
                    v
                )  # e.g. ["noise cancelling","wireless"] → "noise cancelling,wireless"
            else:
                flat[k] = v
        metadatas.append(flat)

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✅  Indexed {len(ids)} products into '{COLLECTION_NAME}' at {DB_PATH}")


if __name__ == "__main__":
    build()
