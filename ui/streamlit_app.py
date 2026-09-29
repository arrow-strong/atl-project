import json
import httpx
import streamlit as st

st.set_page_config(
    page_title="Adaptive Trust Layer",
    page_icon="🛡️",
    layout="wide",
)

AGENT_LABELS = {
    "analyze": "Analyzing the question",
    "plan": "Planning",
    "route": "Choosing the best model",
    "retrieve": "Retrieving documents",
    "generate": "Generating the answer",
    "verify": "Verifying claims against sources",
    "confidence": "Scoring confidence",
}

LABEL_COLOR = {
    "high": "🟢",
    "medium": "🟠",
    "low": "🔴",
}

STATUS_ICON = {
    "supported": "✅",
    "unsupported": "⚠️",
    "contradicted": "❌",
}

EXAMPLES = [
    "What is overfitting?",
    "Compare supervised and unsupervised learning and explain when to use each",
    "Who won the 2010 FIFA World Cup?",
    "Hello, how are you?",
]


# ---------- API call with live progress ----------

def ask_stream(api: str, question: str, on_progress) -> dict:
    with httpx.stream(
        "POST",
        f"{api}/ask/stream",
        json={"question": question},
        timeout=httpx.Timeout(300.0),
    ) as r:
        r.raise_for_status()

        event = None

        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()

            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())

                if event == "progress":
                    on_progress(data["agent"])

                elif event == "final":
                    return data

                elif event == "error":
                    raise RuntimeError(data["detail"])

    raise RuntimeError("Stream ended without a final result")


# ---------- rendering ----------

def render_result(d: dict):
    st.markdown(d["answer"])

    conf = d["confidence"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Confidence",
        f"{conf['score']:.0%}",
        f"{LABEL_COLOR.get(conf['label'], '')} {conf['label']}",
        delta_color="off",
    )

    c2.metric("Model used", d["model_used"])
    c3.metric("Time", f"{d['total_seconds']} s")
    c4.metric("Retries", d["retries"])

    st.progress(min(max(conf["score"], 0.0), 1.0))

    if conf.get("note"):
        st.caption(f"ℹ️ {conf['note']}")

    if d["sources"]:
        with st.expander(f"📚 Sources ({len(d['sources'])})"):
            for i, s in enumerate(d["sources"], 1):
                st.markdown(
                    f"**[{i}] {s['source']}**  ·  "
                    f"similarity {s['similarity']:.2f}"
                )

                st.caption(
                    s["text"][:500]
                    + ("..." if len(s["text"]) > 500 else "")
                )

    elif not d["grounded"]:
        st.caption("No knowledge-base documents were used for this answer.")

    with st.expander("🔍 How ATL answered this"):

        a = d["analysis"]

        st.markdown("**1. Question analysis**")

        st.write(
            f"Type: `{a.get('type')}` · "
            f"Complexity: `{a.get('complexity')}` · "
            f"Needs retrieval: `{a.get('needs_retrieval')}`"
        )

        st.caption(a.get("reason", ""))

        st.markdown("**2. Routing decisions**")

        for t in d["trace"]:
            if t["agent"] == "router":
                st.write(f"→ `{t['model']}`: {t.get('why', 'Routing decision')}")

        st.markdown("**3. Verification**")

        v = d["verification"]

        st.write(
            f"Verdict: `{v.get('verdict')}` · "
            f"Support ratio: `{v.get('support_ratio')}`"
        )

        if v.get("note"):
            st.caption(v["note"])

        for det in v.get("details", []):
            icon = STATUS_ICON.get(det["status"], "•")

            st.write(
                f"{icon} {det['claim']}  \n"
                f"<small>"
                f"entailment {det['entailment']} · "
                f"best passage [{det['best_passage']}]"
                f"</small>",
                unsafe_allow_html=True,
            )

        st.markdown("**4. Confidence signals**")

        st.json(conf["signals"])

        st.markdown("**5. Agent trace**")

        for t in d["trace"]:
            secs = t.get("seconds")

            head = (
                f"`{t['agent']}`"
                + (f"  ({secs}s)" if secs is not None else "")
            )

            with st.expander(head):
                st.json(
                    {
                        k: val
                        for k, val in t.items()
                        if k not in ("agent", "seconds")
                    }
                )


# ---------- sidebar ----------

st.sidebar.title("🛡️ Adaptive Trust Layer")

api = st.sidebar.text_input(
    "API URL",
    "http://127.0.0.1:8000",
)

try:
    h = httpx.get(
        f"{api}/health",
        timeout=3,
    ).json()

    st.sidebar.success(
        f"Backend online · {h['kb_chunks']} chunks"
    )

except Exception:
    st.sidebar.error(
        "Backend not reachable. Start it with: uvicorn app.main:app"
    )


st.sidebar.markdown("**Try an example**")

for ex in EXAMPLES:
    if st.sidebar.button(
        ex,
        use_container_width=True,
    ):
        st.session_state["pending"] = ex


if st.sidebar.button("🗑️ Clear chat"):
    st.session_state["history"] = []


# ---------- main chat ----------

st.title("Adaptive Trust Layer")

st.caption(
    "Every answer is routed, verified against sources, "
    "and scored for confidence."
)

st.session_state.setdefault("history", [])

for item in st.session_state["history"]:

    with st.chat_message("user"):
        st.markdown(item["question"])

    with st.chat_message("assistant"):
        render_result(item["result"])


question = st.chat_input("Ask a question...")

if "pending" in st.session_state:
    question = st.session_state.pop("pending")


if question:

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.status(
                "Working...",
                expanded=True,
            ) as status:

                def on_progress(agent):
                    st.write(
                        f"✔ {AGENT_LABELS.get(agent, agent)}"
                    )

                result = ask_stream(
                    api,
                    question,
                    on_progress,
                )

                status.update(
                    label="Done",
                    state="complete",
                    expanded=False,
                )

            render_result(result)

            st.session_state["history"].append(
                {
                    "question": question,
                    "result": result,
                }
            )

        except Exception as e:
            st.error(f"Request failed: {e}")