from app.llm_client import call_llm
from app.utils import add_trace


def generator(state: dict) -> dict:
    query = state["query"]
    context = state.get("context", [])
    kb_covered = state.get("kb_covered", False)

    if context and kb_covered:
        numbered_context = "\n\n".join(
            f"[{i + 1}] {chunk['text']}"
            for i, chunk in enumerate(context)
        )

        prompt = f"""
Answer the user's question using ONLY the context below.

User question:
{query}

Context:
{numbered_context}

Rules:
- Use only information supported by the context.
- Add citations like [1] or [2] after claims.
- If the context is insufficient, clearly say that it is insufficient.
- Do not invent facts.
"""

        if state.get("verdict", {}).get("unsupported_claims"):
            prompt += f"""

Previous verification found unsupported claims:
{state["verdict"]["unsupported_claims"]}

Revise the answer so that unsupported claims are removed.
"""

        grounded = True

    else:
        prompt = f"""
Answer the following user question:

{query}

There is no sufficiently relevant knowledge-base context available.
Answer without pretending that the knowledge base supports the answer.
If you are uncertain, say so rather than inventing facts.
"""

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