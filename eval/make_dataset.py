import json
import random
import time
from collections import defaultdict

import chromadb
from pydantic import BaseModel

from app.config import CHROMA_DIR, COLLECTION_NAME
from app.utils import call_json


random.seed(42)

GEN_MODEL = "groq-large"
OUTPUT_FILE = "eval/dataset.json"
DELAY_SECONDS = 3


class QA(BaseModel):
    question: str
    answer: str


SIMPLE = """Read the passage and write ONE factual question that can be answered using only this passage, plus a short reference answer (1-2 sentences).

The question must make sense on its own (never say "according to the passage").

Reply ONLY with JSON:
{{"question": "...", "answer": "..."}}

PASSAGE:
{p}
"""


COMPLEX = """Read the two passages and write ONE question that needs information from BOTH (a comparison, relationship, or combination), plus a reference answer (2-3 sentences).

The question must make sense on its own.

Reply ONLY with JSON:
{{"question": "...", "answer": "..."}}

PASSAGE 1:
{a}

PASSAGE 2:
{b}
"""


OUT_OF_KB = [
    ("Who won the 2010 FIFA World Cup?", "Spain"),
    ("What is the capital of Australia?", "Canberra"),
    ("Who wrote the novel Pride and Prejudice?", "Jane Austen"),
    ("What is the chemical symbol for gold?", "Au"),
    ("In which year did India gain independence?", "1947"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    (
        "What is the boiling point of water at sea level in Celsius?",
        "100 degrees Celsius",
    ),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("What is the largest ocean on Earth?", "The Pacific Ocean"),
]


FALSE_PREMISE = [
    (
        "Why did Alan Turing invent the transformer architecture?",
        "False premise: transformers were introduced in 2017 by Vaswani et al., not Turing.",
    ),
    (
        "In what year did Geoffrey Hinton win the Fields Medal for inventing backpropagation?",
        "False premise: Hinton did not win a Fields Medal, and he did not invent backpropagation alone.",
    ),
    (
        "Explain how k-means clustering guarantees finding the global optimum.",
        "False premise: k-means only converges to a local optimum.",
    ),
    (
        "Why does gradient descent always converge in a single step for any neural network?",
        "False premise: it does not; it is an iterative method.",
    ),
    (
        "What did the 1995 paper 'Attention Is All You Need' propose?",
        "False premise: that paper is from 2017.",
    ),
    (
        "Which chapter of our college's syllabus covers quantum backpropagation?",
        "Unanswerable: no such information exists in the knowledge base.",
    ),
    (
        "Why are decision trees always more accurate than random forests?",
        "False premise: random forests usually outperform single trees.",
    ),
    (
        "What is the exact GPU count used to train the first convolutional neural network in 1950?",
        "False premise: CNNs did not exist in 1950 and GPUs were not used.",
    ),
]


def save_items(items):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def load_existing():
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)

        if isinstance(items, list):
            return items

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return []


def generate_json(prompt):
    while True:
        try:
            result = call_json(
                GEN_MODEL,
                prompt,
                QA,
            )

            time.sleep(DELAY_SECONDS)
            return result

        except Exception as e:
            print(f"\nGeneration failed: {e}")
            print("Waiting 15 seconds before retrying...\n")
            time.sleep(15)


def main():
    existing = load_existing()

    print(f"Existing saved questions: {len(existing)}")

    existing_keys = {
        (item["category"], item["question"])
        for item in existing
    }

    col = chromadb.PersistentClient(
        path=CHROMA_DIR
    ).get_collection(COLLECTION_NAME)

    data = col.get(
        include=["documents", "metadatas"]
    )

    chunks = [
        (d, m["source"])
        for d, m in zip(
            data["documents"],
            data["metadatas"],
        )
        if len(d.split()) > 150
    ]

    random.shuffle(chunks)

    items = existing.copy()

    # -----------------------------
    # IN-KB SIMPLE
    # -----------------------------

    per_source = defaultdict(int)
    picked = []

    for text, src in chunks:
        if per_source[src] < 2:
            per_source[src] += 1
            picked.append((text, src))

        if len(picked) == 40:
            break

    for text, src in picked:

        prompt = SIMPLE.format(p=text)

        print(f"Generating simple {len(items) + 1}...")

        qa = generate_json(prompt)

        key = ("in_kb_simple", qa.question)

        if key not in existing_keys:
            items.append(
                {
                    "category": "in_kb_simple",
                    "question": qa.question,
                    "reference": qa.answer,
                    "expected": "answer",
                    "source": src,
                }
            )

            existing_keys.add(key)
            save_items(items)

    # -----------------------------
    # IN-KB COMPLEX
    # -----------------------------

    for _ in range(15):

        while True:
            (a, sa), (b, sb) = random.sample(chunks, 2)

            if sa != sb:
                break

        prompt = COMPLEX.format(
            a=a,
            b=b,
        )

        print(f"Generating complex {len(items) + 1}...")

        qa = generate_json(prompt)

        key = ("in_kb_complex", qa.question)

        if key not in existing_keys:
            items.append(
                {
                    "category": "in_kb_complex",
                    "question": qa.question,
                    "reference": qa.answer,
                    "expected": "answer",
                    "source": f"{sa} + {sb}",
                }
            )

            existing_keys.add(key)
            save_items(items)

    # -----------------------------
    # OUT OF KB
    # -----------------------------

    for q, a in OUT_OF_KB:

        key = ("out_of_kb", q)

        if key not in existing_keys:
            items.append(
                {
                    "category": "out_of_kb",
                    "question": q,
                    "reference": a,
                    "expected": "answer",
                    "source": "-",
                }
            )

            existing_keys.add(key)

    # -----------------------------
    # FALSE PREMISE
    # -----------------------------

    for q, a in FALSE_PREMISE:

        key = ("false_premise", q)

        if key not in existing_keys:
            items.append(
                {
                    "category": "false_premise",
                    "question": q,
                    "reference": a,
                    "expected": "reject",
                    "source": "-",
                }
            )

            existing_keys.add(key)

    # -----------------------------
    # IDs + SPLITS
    # -----------------------------

    for i, item in enumerate(items):
        item["id"] = f"q{i:03d}"
        item["split"] = "dev" if i % 2 == 0 else "test"

    save_items(items)

    print()
    print(f"Wrote {len(items)} questions")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()