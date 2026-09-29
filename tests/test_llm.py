from app.llm_client import call_llm

for m in ["local-small", "groq-medium", "groq-large"]:
    out = call_llm(
        m,
        "What is the capital of India? Answer in one sentence."
    )

    print(f"[{m}] ({out['latency_s']}s): {out['text']}\n")