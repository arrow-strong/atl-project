from app.agents.analyzer import analyzer
from app.agents.planner import planner
from app.agents.router import router
from app.agents.retriever import retriever
from app.agents.generator import generator
from app.agents.verifier import verifier
from app.agents.confidence import confidence
from app.config import MAX_RETRIES


def run(query: str) -> dict:
    state = {
        "query": query,
        "retries": 0,
        "trace": [],
    }

    state.update(analyzer(state))
    state.update(planner(state))
    state.update(router(state))

    if state["analysis"]["needs_retrieval"]:
        state.update(retriever(state))

    state.update(generator(state))
    state.update(verifier(state))

    while (
        state["verdict"]["verdict"] == "fail"
        and state["retries"] <= MAX_RETRIES - 1
    ):
        state.update(router(state))
        state.update(generator(state))
        state.update(verifier(state))

    state.update(confidence(state))

    return state


queries = [
    "Hello, how are you?",
    "What is overfitting?",
    "Compare supervised and unsupervised learning and explain when to use each",
    "Who won the 2010 FIFA World Cup?",
]


for query in queries:
    print("\n" + "=" * 70)

    s = run(query)

    print("Query:", query)
    print("Model:", s["model"])
    print("Answer:", s["answer"])
    print(
        "Verdict:",
        s["verdict"]["verdict"],
        "| support:",
        s["verdict"].get("support_ratio"),
    )
    print(
        "Confidence:",
        s["confidence"],
        s["confidence_label"],
        s["confidence_signals"],
        s["confidence_note"],
    )