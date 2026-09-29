import time
from pathlib import Path
import httpx

TITLES = [
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Transformer (deep learning architecture)",
    "Large language model",
    "Retrieval-augmented generation",
    "Natural language processing",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Convolutional neural network",
    "Recurrent neural network",
    "Gradient descent",
    "Backpropagation",
    "Overfitting",
    "Support vector machine",
    "Decision tree",
    "Random forest",
    "K-means clustering",
    "Word embedding",
]

OUT = Path("data/docs")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "ATL-student-project/0.1 (student project; contact: kamakshi@gmail.com)"
}

for title in TITLES:
    r = httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        headers=HEADERS,
        timeout=30,
    )

    r.raise_for_status()

    pages = r.json()["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")

    if len(text) < 500:
        print(f"skipped (too short/missing): {title}")
        continue

    fname = title.replace(" ", "_").replace("/", "_") + ".txt"

    (OUT / fname).write_text(
        text,
        encoding="utf-8"
    )

    print(f"saved {fname} ({len(text)} chars)")

    time.sleep(0.5)