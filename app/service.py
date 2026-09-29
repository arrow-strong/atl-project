from app.schemas import AskResponse, Confidence, Source


def build_response(state: dict, total_seconds: float) -> AskResponse:
    return AskResponse(
        question=state["query"],
        answer=state.get("answer", ""),
        confidence=Confidence(
            score=state.get("confidence", 0.0),
            label=state.get("confidence_label", "low"),
            signals=state.get("confidence_signals", {}),
            note=state.get("confidence_note", ""),
        ),
        model_used=state.get("model", "unknown"),
        grounded=state.get("grounded", False),
        verification=state.get("verdict", {}),
        sources=[
            Source(
                id=c["id"],
                source=c["source"],
                text=c["text"],
                similarity=c["similarity"],
            )
            for c in (state.get("context") or [])
        ] if state.get("grounded") else [],
        analysis=state.get("analysis", {}),
        retries=state.get("retries", 0),
        trace=state.get("trace", []),
        total_seconds=round(total_seconds, 2),
    )