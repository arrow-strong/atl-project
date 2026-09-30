import time

import httpx

from app.config import MODELS, GROQ_API_KEY, OLLAMA_URL

CALL_LOG = []


def _call_ollama(model_id: str, prompt: str, temperature: float) -> str:
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def _call_groq(model_id: str, prompt: str, temperature: float) -> str:
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_llm(
    model_name: str,
    prompt: str,
    temperature: float = 0.2,
    retries: int = 3,
) -> dict:
    """Single entry point for every model. Returns text + latency for logging."""

    cfg = MODELS[model_name]
    fn = _call_ollama if cfg["provider"] == "ollama" else _call_groq

    last_err = None

    for attempt in range(retries):
        try:
            start = time.time()

            text = fn(
                cfg["model_id"],
                prompt,
                temperature,
            )

            result = {
                "text": text,
                "model": model_name,
                "latency_s": round(time.time() - start, 2),
            }

            CALL_LOG.append({
                "model": model_name,
                "latency_s": result["latency_s"],
            })

            return result

        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"{model_name} failed after {retries} tries: {last_err}"
    )