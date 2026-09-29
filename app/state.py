from typing import TypedDict, List


class ATLState(TypedDict, total=False):
    # input
    query: str

    # analyzer / planner / router
    analysis: dict
    plan: List[str]
    model: str

    # retriever
    context: List[dict]
    kb_covered: bool

    # generator
    answer: str
    grounded: bool

    # verifier
    verdict: dict
    retries: int

    # confidence
    confidence: float
    confidence_label: str
    confidence_signals: dict
    confidence_note: str

    # transparency
    trace: List[dict]