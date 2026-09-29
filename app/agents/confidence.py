import time
import numpy as np

from app.config import (
    CONF_WEIGHTS,
    ENABLE_AGREEMENT,
    KB_SIM_THRESHOLD,
    UNGROUNDED_CAP,
)
from app.llm_client import call_llm
from app.agents.generator import GROUNDED_PROMPT, OPEN_PROMPT
from app.retrieval import embed
from app.utils import add_trace


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _retrieval_signal(state: dict):
    if not state.get("grounded"):
        return None

    top = max(
        c["similarity"]
        for c in state["context"]
    )

    return _clip(
        (top - KB_SIM_THRESHOLD)
        / (0.8 - KB_SIM_THRESHOLD)
    )


def _agreement_signal(state: dict):
    alt_model = (
        "groq-medium"
        if state["model"] != "groq-medium"
        else "groq-large"
    )

    if state.get("grounded"):
        ctx = "\n\n".join(
            f"[{i + 1}] ({c['source']}) {c['text']}"
            for i, c in enumerate(state["context"])
        )

        prompt = GROUNDED_PROMPT.format(
            feedback="",
            context=ctx,
            query=state["query"],
        )
    else:
        prompt = OPEN_PROMPT.format(
            feedback="",
            query=state["query"],
        )

    alt = call_llm(
        alt_model,
        prompt,
        temperature=0.2,
    )["text"]

    a, b = embed([
        state["answer"],
        alt,
    ])

    return _clip(float(np.dot(a, b)))


def _label(score: float) -> str:
    return (
        "high"
        if score >= 0.75
        else "medium"
        if score >= 0.5
        else "low"
    )


def confidence(state: dict) -> dict:
    start = time.time()
    verdict = state.get("verdict", {})
    qtype = state["analysis"]["type"]

    if qtype == "chitchat":
        return {
            "confidence": 0.9,
            "confidence_label": "high",
            "confidence_signals": {},
            "confidence_note": "conversational reply, no factual claims",
            "trace": add_trace(
                state,
                "confidence",
                score=0.9,
                note="chitchat",
            ),
        }

    signals = {
        "support": verdict.get("support_ratio"),
        "retrieval": _retrieval_signal(state),
        "agreement": (
            _agreement_signal(state)
            if ENABLE_AGREEMENT
            else None
        ),
    }

    available = {
        k: v
        for k, v in signals.items()
        if v is not None
    }

    if available:
        total_w = sum(
            CONF_WEIGHTS[k]
            for k in available
        )

        score = sum(
            CONF_WEIGHTS[k] * v
            for k, v in available.items()
        ) / total_w
    else:
        score = 0.3

    notes = []

    if not state.get("grounded"):
        score = min(
            score,
            UNGROUNDED_CAP,
        )
        notes.append(
            "no supporting documents found; score capped"
        )

    if verdict.get("verdict") == "abstained":
        score = min(score, 0.3)
        notes.append(
            "model abstained; answer is honest but not informative"
        )

    if verdict.get("verdict") == "fail":
        notes.append(
            "verification failed after retries"
        )

    score = round(score, 3)

    return {
        "confidence": score,
        "confidence_label": _label(score),
        "confidence_signals": {
            k: (
                round(v, 3)
                if v is not None
                else None
            )
            for k, v in signals.items()
        },
        "confidence_note": "; ".join(notes),
        "trace": add_trace(
            state,
            "confidence",
            score=score,
            label=_label(score),
            signals=signals,
            seconds=round(
                time.time() - start,
                2,
            ),
        ),
    }