from langgraph.graph import StateGraph, END

from app.config import MAX_RETRIES
from app.state import ATLState
from app.agents.analyzer import analyzer
from app.agents.planner import planner
from app.agents.router import router
from app.agents.retriever import retriever
from app.agents.generator import generator
from app.agents.verifier import verifier
from app.agents.confidence import confidence


def after_router(state: ATLState) -> str:
    first_pass = state.get("retries", 0) == 0

    if state["analysis"]["needs_retrieval"] and first_pass:
        return "retrieve"

    return "generate"


def after_verify(state: ATLState) -> str:
    if (
        state["verdict"]["verdict"] == "fail"
        and state.get("retries", 0) < MAX_RETRIES
    ):
        return "route"

    return "confidence"


def build_graph():
    g = StateGraph(ATLState)

    g.add_node("analyze", analyzer)
    g.add_node("plan", planner)
    g.add_node("route", router)
    g.add_node("retrieve", retriever)
    g.add_node("generate", generator)
    g.add_node("verify", verifier)
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


atl = build_graph()


def run_atl(query: str) -> dict:
    return atl.invoke({
        "query": query,
        "retries": 0,
        "trace": [],
    })


def warm_up():
    from app.retrieval import _init
    from app.agents.verifier import _load_nli

    _init()
    _load_nli()