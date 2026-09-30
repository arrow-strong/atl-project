import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.graph import warm_up
from app.systems import SYSTEMS, timed
from app.utils import call_json


JUDGE_MODEL = "groq-large"

OUT = Path("eval/results")
OUT.mkdir(parents=True, exist_ok=True)


class Judgement(BaseModel):
    verdict: Literal[
        "correct",
        "partially_correct",
        "incorrect",
        "abstained",
    ]
    reason: str


JUDGE_PROMPT = """You are grading an AI answer against a reference.

QUESTION: {q}

EXPECTED BEHAVIOUR: {expected}

REFERENCE: {ref}

AI ANSWER: {a}

Rules:

- If expected behaviour is "answer":
  correct = the key facts match the reference.
  partially_correct = some key facts right, others missing.
  incorrect = contradicts the reference or states wrong facts.
  abstained = says it does not know or the information is unavailable.

- If expected behaviour is "reject" (false premise or unanswerable):
  correct = the answer points out the false premise or says it cannot be answered.
  incorrect = it plays along and invents an answer.
  Never use partially_correct or abstained here.

- Extra true detail that is not in the reference is fine.

Reply ONLY with JSON:

{{"verdict": "correct|partially_correct|incorrect|abstained", "reason": "one sentence"}}
"""


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--systems",
        nargs="+",
        required=True,
        choices=list(SYSTEMS),
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N questions",
    )

    args = ap.parse_args()

    with open(
        "eval/dataset.json",
        encoding="utf-8",
    ) as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    warm_up()

    for name in args.systems:

        fn = SYSTEMS[name]()

        path = OUT / f"{name}.jsonl"

        if path.exists():
            done = {
                json.loads(line)["id"]
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }
        else:
            done = set()

        print(
            f"\n=== {name}: "
            f"{len(dataset) - len(done)} to run ==="
        )

        for item in dataset:

            if item["id"] in done:
                continue

            try:

                res = timed(
                    fn,
                    item["question"],
                )

                judge = call_json(
                    JUDGE_MODEL,
                    JUDGE_PROMPT.format(
                        q=item["question"],
                        expected=item["expected"],
                        ref=item["reference"],
                        a=res["answer"],
                    ),
                    Judgement,
                )

            except Exception as e:

                print(
                    f"  {item['id']} failed, "
                    f"will retry on next run: {e}"
                )

                continue

            res.update(
                {
                    "id": item["id"],
                    "verdict": judge.verdict,
                    "judge_reason": judge.reason,
                }
            )

            with open(
                path,
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        res,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"  {item['id']} "
                f"[{item['category']}] "
                f"{judge.verdict} "
                f"({res['seconds']}s)"
            )


if __name__ == "__main__":
    main()