import json

from app.agents.retriever import retriever
from app.agents.verifier import verifier


query = "What is overfitting?"

state = {
    "query": query,
    "plan": {
        "sub_questions": [query]
    },
    "retries": 0,
    "trace": [],
}

state.update(retriever(state))

state["grounded"] = state["kb_covered"]

state["answer"] = (
    "Overfitting happens when a model learns the training data too closely, "
    "including noise, so it performs poorly on new data. "
    "The term was invented by Alan Turing in 1950."
)

state.update(verifier(state))

print(json.dumps(state["verdict"], indent=2))