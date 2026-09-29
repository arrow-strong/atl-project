import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBED_MODEL, COLLECTION_NAME


_model = None
_collection = None


def _init():
    global _model, _collection

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)

        client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        _collection = client.get_collection(
            COLLECTION_NAME
        )


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return the top-k chunks with source and cosine similarity."""

    _init()

    q_emb = _model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    res = _collection.query(
        query_embeddings=q_emb,
        n_results=k
    )

    results = []

    for i in range(len(res["ids"][0])):
        results.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "chunk_index": res["metadatas"][0][i]["chunk_index"],
            "similarity": round(
                1 - res["distances"][0][i],
                4
            ),
        })

    return results


def embed(texts: list[str]):
    """Normalized embeddings as a numpy array (cosine = dot product)."""
    _init()
    return _model.encode(texts, normalize_embeddings=True)