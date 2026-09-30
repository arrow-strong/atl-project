import time

from app.ablation import build_variant
from app.agents.generator import GROUNDED_PROMPT, OPEN_PROMPT
from app.graph import run_atl
from app.llm_client import call_llm, CALL_LOG
from app.retrieval import retrieve


COST = {
    "local-small": 0.0,
    "groq-medium": 1.0,
    "groq-large": 9.0,
}


def _plain(model):
    def f(q):
        out = call_llm(
            model,
            OPEN_PROMPT.format(
                feedback="",
                query=q,
            ),
        )

        return {
            "answer": out["text"].strip(),
            "model": model,
        }

    return f


def _rag(model, k=4):
    def f(q):
        chunks = retrieve(q, k=k)

        ctx = "\n\n".join(
            f"[{i+1}] ({c['source']}) {c['text']}"
            for i, c in enumerate(chunks)
        )

        out = call_llm(
            model,
            GROUNDED_PROMPT.format(
                feedback="",
                context=ctx,
                query=q,
            ),
        )

        return {
            "answer": out["text"].strip(),
            "model": model,
        }

    return f


def _atl(graph=None):

    def f(q):

        if graph:
            state = graph.invoke(
                {
                    "query": q,
                    "retries": 0,
                    "trace": [],
                }
            )
        else:
            state = run_atl(q)

        return {
            "answer": state["answer"],
            "model": state.get("model"),
            "confidence": state.get("confidence"),
            "signals": state.get("confidence_signals"),
            "retries": state.get("retries", 0),
            "grounded": state.get("grounded", False),
            "verdict_internal": state.get(
                "verdict",
                {},
            ).get("verdict"),
        }

    return f


SYSTEMS = {
    "A_small": lambda: _plain("local-small"),

    "B_large": lambda: _plain("groq-large"),

    "C_rag_medium": lambda: _rag("groq-medium"),

    "D_atl": lambda: _atl(),

    "D1_no_verifier": lambda: _atl(
        build_variant(
            use_verifier=False
        )
    ),

    "D2_fixed_large": lambda: _atl(
        build_variant(
            use_router=False,
            fixed_model="groq-large",
        )
    ),

    "D3_fixed_medium": lambda: _atl(
        build_variant(
            use_router=False,
            fixed_model="groq-medium",
        )
    ),
}


def timed(fn, q):

    CALL_LOG.clear()

    start = time.time()

    result = fn(q)

    result["seconds"] = round(
        time.time() - start,
        2,
    )

    result["llm_calls"] = len(CALL_LOG)

    result["cost_units"] = sum(
        COST[c["model"]]
        for c in CALL_LOG
    )

    return result