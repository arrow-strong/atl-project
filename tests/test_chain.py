from app.agents.analyzer import analyzer
from app.agents.planner import planner
from app.agents.router import router
from app.agents.retriever import retriever
from app.agents.generator import generator


QUERIES = [
    "Hello, how are you?",
    "What is overfitting?",
    "Compare supervised and unsupervised learning and explain when to use each",
    "Who won the 2010 FIFA World Cup?",
]


for query in QUERIES:
    print("\n" + "=" * 70)
    print("QUERY:", query)

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

    print("ANALYSIS:", state["analysis"])
    print("PLAN:", state["plan"])
    print("MODEL:", state["model"])
    print("KB COVERED:", state.get("kb_covered"))
    print("ANSWER:", state["answer"])
    print("TRACE:", state["trace"])