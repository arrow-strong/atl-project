from typing import Literal

from pydantic import BaseModel

from app.config import ANALYZER_MODEL
from app.utils import add_trace, call_json


class Analysis(BaseModel):
    type: Literal[
        "factual",
        "explanation",
        "reasoning",
        "math",
        "coding",
        "opinion",
        "chitchat",
    ]
    complexity: Literal["low", "medium", "high"]
    needs_retrieval: bool
    reason: str


def analyzer(state: dict) -> dict:
    query = state["query"]

    prompt = f"""
Classify the following user query.

Query:
{query}

Return ONLY a JSON object with these fields:

{{
  "type": "factual|explanation|reasoning|math|coding|opinion|chitchat",
  "complexity": "low|medium|high",
  "needs_retrieval": true,
  "reason": "brief explanation"
}}

Rules:
- factual questions about the knowledge base usually need retrieval
- explanations about knowledge-base topics usually need retrieval
- greetings/chitchat do not need retrieval
- math, coding, and reasoning questions may need retrieval depending on the query
"""

    analysis = call_json(
        ANALYZER_MODEL,
        prompt,
        Analysis,
    )

    return {
        "analysis": analysis.model_dump(),
        "trace": add_trace(
            state,
            "analyzer",
            type=analysis.type,
            complexity=analysis.complexity,
            needs_retrieval=analysis.needs_retrieval,
        ),
    }