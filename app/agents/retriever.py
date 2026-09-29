from app.config import KB_SIM_THRESHOLD, MAX_CONTEXT_CHUNKS, TOP_K
from app.retrieval import retrieve
from app.utils import add_trace


def retriever(state: dict) -> dict:
    sub_questions = state["plan"]["sub_questions"]

    all_chunks = {}

    for sub_q in sub_questions:
        results = retrieve(sub_q, k=TOP_K)

        for chunk in results:
            chunk_id = chunk["id"]

            if (
                chunk_id not in all_chunks
                or chunk["similarity"] > all_chunks[chunk_id]["similarity"]
            ):
                all_chunks[chunk_id] = chunk

    chunks = sorted(
        all_chunks.values(),
        key=lambda x: x["similarity"],
        reverse=True,
    )

    chunks = chunks[:MAX_CONTEXT_CHUNKS]

    top_sim = chunks[0]["similarity"] if chunks else 0.0
    covered = top_sim >= KB_SIM_THRESHOLD

    return {
        "context": chunks,
        "kb_covered": covered,
        "trace": add_trace(
            state,
            "retriever",
            chunks=len(chunks),
            top_similarity=top_sim,
            kb_covered=covered,
        ),
    }