import re
import time
from typing import List

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import ANALYZER_MODEL, NLI_MODEL, SUPPORT_THRESHOLD, PASS_RATIO
from app.utils import call_json, add_trace


class Claims(BaseModel):
    claims: List[str]


CLAIM_PROMPT = """Break the ANSWER into atomic factual claims.
Rules:
- Each claim must be a single, self-contained statement (replace pronouns like "it" or "they" with the actual subject).
- Skip opinions, hedges, greetings, and statements saying information is missing or unavailable.
- Maximum 8 claims. If there are no factual claims, return an empty list.
Reply with ONLY JSON: {{"claims": ["...", "..."]}}

QUESTION: {query}
ANSWER: {answer}"""


_tok = _nli = None
_ENT = _CON = None


def _load_nli():
    global _tok, _nli, _ENT, _CON

    if _nli is None:
        _tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL
        ).eval()

        labels = {
            int(k): v.lower()
            for k, v in _nli.config.id2label.items()
        }

        _ENT = next(
            i for i, l in labels.items()
            if "entail" in l
        )

        _CON = next(
            i for i, l in labels.items()
            if "contra" in l
        )


def _windows(
    text: str,
    size: int = 100,
    stride: int = 60,
) -> List[str]:
    words = text.split()

    if len(words) <= size:
        return [text]

    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words) - 20, stride)
    ]


@torch.no_grad()
def _nli_scores(
    premises: List[str],
    claim: str,
    batch: int = 16,
):
    _load_nli()

    ent, con = [], []

    for i in range(0, len(premises), batch):
        part = premises[i:i + batch]

        enc = _tok(
            [claim] * len(part),
            part,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )

        probs = torch.softmax(
            _nli(**enc).logits,
            dim=-1,
        )

        ent += probs[:, _ENT].tolist()
        con += probs[:, _CON].tolist()

    return ent, con


def verifier(state: dict) -> dict:
    start = time.time()
    retries = state.get("retries", 0)
    answer = state["answer"]

    if not state.get("grounded"):
        verdict = {
            "verdict": "unverified",
            "support_ratio": None,
            "details": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "note": "answer was not grounded in retrieved documents",
        }

        return {
            "verdict": verdict,
            "trace": add_trace(
                state,
                "verifier",
                result="unverified (no context)",
                seconds=round(time.time() - start, 2),
            ),
        }

    clean_answer = re.sub(r"\[\d+\]", "", answer)

    claims = call_json(
        ANALYZER_MODEL,
        CLAIM_PROMPT.format(
            query=state["query"],
            answer=clean_answer,
        ),
        Claims,
    ).claims

    if not claims:
        verdict = {
            "verdict": "abstained",
            "support_ratio": None,
            "details": [],
            "supported_claims": [],
            "unsupported_claims": [],
            "note": "answer made no factual claims (likely an honest abstention)",
        }

        return {
            "verdict": verdict,
            "trace": add_trace(
                state,
                "verifier",
                result="abstained",
                seconds=round(time.time() - start, 2),
            ),
        }

    premises, owner = [], []

    for idx, chunk in enumerate(state["context"], start=1):
        for window in _windows(chunk["text"]):
            premises.append(window)
            owner.append(idx)

    details = []

    for claim in claims:
        ent, con = _nli_scores(premises, claim)

        best = max(
            range(len(ent)),
            key=lambda i: ent[i],
        )

        if ent[best] >= SUPPORT_THRESHOLD:
            status = "supported"
        else:
            status = "unsupported"

        details.append({
            "claim": claim,
            "status": status,
            "entailment": round(ent[best], 3),
            "best_passage": owner[best],
        })

    supported = [
        d["claim"]
        for d in details
        if d["status"] == "supported"
    ]

    bad = [
        d["claim"]
        for d in details
        if d["status"] != "supported"
    ]

    contradicted = any(
        d["status"] == "contradicted"
        for d in details
    )

    ratio = len(supported) / len(details)
    passed = ratio >= PASS_RATIO and not contradicted

    verdict = {
        "verdict": "pass" if passed else "fail",
        "support_ratio": round(ratio, 3),
        "supported_claims": supported,
        "unsupported_claims": bad,
        "details": details,
    }

    return {
        "verdict": verdict,
        "retries": retries + (0 if passed else 1),
        "trace": add_trace(
            state,
            "verifier",
            result=verdict["verdict"],
            support_ratio=verdict["support_ratio"],
            claims=len(details),
            seconds=round(time.time() - start, 2),
        ),
    }