from langgraph.graph import StateGraph, END

from app.graph import after_router, after_verify
from app.state import ATLState
from app.utils import add_trace
from app.agents.analyzer import analyzer
from app.agents.planner import planner
from app.agents.router import router
from app.agents.retriever import retriever
from app.agents.generator import generator
from app.agents.verifier import verifier
from app.agents.confidence import confidence


def build_variant(use_router=True, use_verifier=True, fixed_model="groq-large"):

    def fixed_router(state):
        return {
            "model": fixed_model,
            "trace": add_trace(
                state,
                "router",
                model=fixed_model,
                why="ablation: fixed model",
            ),
        }

    def skip_verifier(state):
        v = {
            "verdict": "skipped",
            "support_ratio": None,
            "details": [],
            "supported_claims": [],
            "unsupported_claims": [],
        }

        return {
            "verdict": v,
            "trace": add_trace(
                state,
                "verifier",
                result="skipped (ablation)",
            ),
        }

    g = StateGraph(ATLState)

    g.add_node("analyze", analyzer)
    g.add_node("plan", planner)
    g.add_node("route", router if use_router else fixed_router)
    g.add_node("retrieve", retriever)
    g.add_node("generate", generator)
    g.add_node(
        "verify",
        verifier if use_verifier else skip_verifier,
    )
    g.add_node("confidence", confidence)

    g.set_entry_point("analyze")

    g.add_edge("analyze", "plan")
    g.add_edge("plan", "route")

    g.add_conditional_edges(
        "route",
        after_router,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    g.add_edge("retrieve", "generate")

    g.add_edge("generate", "verify")

    g.add_conditional_edges(
        "verify",
        after_verify,
        {
            "route": "route",
            "confidence": "confidence",
        },
    )

    g.add_edge("confidence", END)

    return g.compile()