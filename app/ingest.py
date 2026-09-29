import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCS_DIR,
    CHROMA_DIR,
    EMBED_MODEL,
    COLLECTION_NAME,
    CHUNK_WORDS,
    CHUNK_OVERLAP,
)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(
            (page.extract_text() or "")
            for page in reader.pages
        )

    return path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def clean(text: str) -> str:
    text = re.sub(r"={2,}.*?={2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
):
    words = text.split()
    step = size - overlap
    chunks = []

    for start in range(0, len(words), step):
        piece = words[start:start + size]

        if len(piece) < 40:
            break

        chunks.append(" ".join(piece))

    return chunks


def main():
    files = [
        p
        for p in Path(DOCS_DIR).glob("**/*")
        if p.suffix.lower() in {".pdf", ".txt", ".md"}
    ]

    if not files:
        raise SystemExit(
            f"No documents found in {DOCS_DIR}"
        )

    print(f"Found {len(files)} documents")

    model = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Start fresh so re-running doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0

    for path in files:
        text = clean(load_document(path))
        chunks = chunk_text(text)

        if not chunks:
            print(
                f"  skipped (no usable text): {path.name}"
            )
            continue

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [
            f"{path.stem}::{i}"
            for i in range(len(chunks))
        ]

        metas = [
            {
                "source": path.name,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        col.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metas,
        )

        total += len(chunks)

        print(
            f"  {path.name}: {len(chunks)} chunks"
        )

    print(
        f"\nDone. {total} chunks stored in {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()