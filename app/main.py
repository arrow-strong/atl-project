import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import MODELS
from app.graph import atl, run_atl, warm_up
from app.schemas import AskRequest, AskResponse
from app.service import build_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding + NLI models...")
    warm_up()
    print("ATL ready.")
    yield


app = FastAPI(
    title="Adaptive Trust Layer",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    from app import retrieval

    return {
        "status": "ok",
        "kb_chunks": retrieval._collection.count(),
    }


@app.get("/models")
def models():
    return {
        name: {
            "provider": m["provider"],
            "tier": m["tier"],
            "model_id": m["model_id"],
        }
        for name, m in MODELS.items()
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.time()

    try:
        state = run_atl(req.question.strip())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {e}",
        )

    return build_response(
        state,
        time.time() - start,
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    def events():
        start = time.time()

        state = {
            "query": req.question.strip(),
            "retries": 0,
            "trace": [],
        }

        try:
            for update in atl.stream(
                state,
                stream_mode="updates",
            ):
                node, delta = next(iter(update.items()))

                state.update(delta)

                yield (
                    f"event: progress\n"
                    f"data: {json.dumps({'agent': node})}\n\n"
                )

            final = build_response(
                state,
                time.time() - start,
            )

            yield (
                f"event: final\n"
                f"data: {final.model_dump_json()}\n\n"
            )

        except Exception as e:
            yield (
                f"event: error\n"
                f"data: {json.dumps({'detail': str(e)})}\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
    )