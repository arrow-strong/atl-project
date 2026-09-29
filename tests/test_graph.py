from app.graph import run_atl, warm_up


warm_up()

TESTS = [
    "Hello, how are you?",
    "What is overfitting?",
    "Compare supervised and unsupervised learning and explain when to use each",
    "Who won the 2010 FIFA World Cup?",
]


for q in TESTS:
    s = run_atl(q)

    print("=" * 80)
    print("Q:", q)
    print("Path:", " → ".join(t["agent"] for t in s["trace"]))
    print(
        "Model:",
        s["model"],
        "| retries:",
        s["retries"],
        "| grounded:",
        s.get("grounded"),
    )
    print(
        "Verdict:",
        s["verdict"]["verdict"],
        "| support:",
        s["verdict"].get("support_ratio"),
    )
    print(
        "Confidence:",
        s["confidence"],
        s["confidence_label"],
        s["confidence_note"],
    )
    print("Answer:", s["answer"][:300])