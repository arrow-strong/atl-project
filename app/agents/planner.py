from typing import List

from pydantic import BaseModel

from app.utils import add_trace, call_json


class Plan(BaseModel):
    sub_questions: List[str]


def planner(state: dict) -> dict:
    query = state["query"]
    complexity = state["analysis"]["complexity"]

    # Simple queries don't need planning
    if complexity != "high":
        plan = Plan(sub_questions=[query])

        return {
            "plan": plan.model_dump(),
            "trace": add_trace(
                state,
                "planner",
                sub_questions=plan.sub_questions,
            ),
        }

    prompt = f"""
Break the following complex user query into 2 to 4
smaller questions that can be answered independently.

Query:
{query}

Return ONLY a JSON object:

{{
  "sub_questions": [
    "question 1",
    "question 2"
  ]
}}
"""

    plan = call_json(
        state["model"],
        prompt,
        Plan,
    )

    return {
        "plan": plan.model_dump(),
        "trace": add_trace(
            state,
            "planner",
            sub_questions=plan.sub_questions,
        ),
    }