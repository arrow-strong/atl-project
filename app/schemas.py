from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)


class Source(BaseModel):
    id: str
    source: str
    text: str
    similarity: float


class Confidence(BaseModel):
    score: float
    label: str
    signals: dict
    note: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: Confidence
    model_used: str
    grounded: bool
    verification: dict
    sources: List[Source]
    analysis: dict
    retries: int
    trace: List[dict]
    total_seconds: float