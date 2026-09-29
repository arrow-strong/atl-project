from app.llm_client import call_llm
from app.utils import add_trace


GROUNDED_PROMPT = """
Answer the user's question using ONLY the context below.

Context:
{context}

Question:
{query}

Rules:
- Use only information supported by the context.
- Add citations like [1] or [2] after claims.
- If the context is insufficient, say so.
- Do not invent facts.

{feedback}
"""


OPEN_PROMPT = """
Answer the user's question:

{query}

There is no sufficiently relevant knowledge-base context available.
Do not pretend the knowledge base supports the answer.
If uncertain, say so rather than inventing facts.

{feedback}
"""


def generator(state: dict) -> dict:
    query = state["query"]
    context = state.get("context", [])
    kb_covered = state.get("kb_covered", False)

    feedback = ""

    if state.get("verdict", {}).get("unsupported_claims"):
        feedback = f"""
Previous verification found unsupported claims:
{state["verdict"]["unsupported_claims"]}

Remove or correct those unsupported claims.
"""

    if context and kb_covered:
        numbered_context = "\n\n".join(
            f"[{i + 1}] {chunk['text']}"
            for i, chunk in enumerate(context)
        )

        prompt = GROUNDED_PROMPT.format(
            context=numbered_context,
            query=query,
            feedback=feedback,
        )

        grounded = True

    else:
        prompt = OPEN_PROMPT.format(
            query=query,
            feedback=feedback,
        )

        grounded = False

    out = call_llm(
        state["model"],
        prompt,
        temperature=0.2,
    )

    return {
        "answer": out["text"],
        "grounded": grounded,
        "trace": add_trace(
            state,
            "generator",
            grounded=grounded,
        ),
    }