from app.config import TIER_ORDER
from app.utils import add_trace


BASE_TIER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

HARD_TYPES = {
    "math",
    "coding",
    "reasoning",
}


def router(state: dict) -> dict:
    analysis = state["analysis"]

    idx = BASE_TIER[analysis["complexity"]]

    # Hard problem types get one extra tier
    if analysis["type"] in HARD_TYPES:
        idx = min(idx + 1, len(TIER_ORDER) - 1)

    # Escalate after failed verification/retries
    retries = state.get("retries", 0)

    if retries > 0:
        idx = min(idx + retries, len(TIER_ORDER) - 1)

    model = TIER_ORDER[idx]

    return {
        "model": model,
        "trace": add_trace(
            state,
            "router",
            model=model,
        ),
    }