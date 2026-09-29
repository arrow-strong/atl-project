import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm_client import call_llm


T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("no JSON object found")

    return json.loads(text[start:end + 1])


def call_json(
    model: str,
    prompt: str,
    schema: Type[T],
    retries: int = 2,
) -> T:
    last_err = None

    for _ in range(retries + 1):
        out = call_llm(
            model,
            prompt,
            temperature=0.0,
        )

        try:
            return schema(**extract_json(out["text"]))

        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last_err = e

            prompt += (
                "\n\nYour previous reply was not valid JSON "
                "for the required schema. Reply with ONLY the JSON object."
            )

    raise RuntimeError(
        f"Could not get valid JSON: {last_err}"
    )


def add_trace(state: dict, agent: str, **info) -> list:
    return state.get("trace", []) + [
        {
            "agent": agent,
            **info,
        }
    ]