"""
ContextChain — FastAPI Backend v4
Streaming SSE + model info endpoint.
"""

import os
import sys
import json
import asyncio
import uuid

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pipeline import run_pipeline_streaming

app = FastAPI(
    title="ContextChain API",
    description="Episodic Memory Objects as a handoff primitive in multi-agent workflows.",
    version="4.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class ProcurementRequest(BaseModel):
    raw_input: str
    task_id: str | None = None


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api/models")
def model_info():
    """Which model+platform each agent uses. Shown in UI so judges see the setup."""
    from core.model_config import AGENT_MODELS
    result = {}
    for agent_id, cfg in AGENT_MODELS.items():
        key_set = bool(os.getenv(cfg["api_key_env"], "").strip())
        result[agent_id] = {
            "platform":       cfg["platform"] if key_set else "OpenAI (fallback)",
            "model":          cfg["model"]    if key_set else cfg["fallback_model"],
            "key_configured": key_set,
            "reason":         cfg["reason"],
        }
    return result


@app.post("/api/run/stream")
async def run_stream(req: ProcurementRequest):
    """SSE streaming endpoint — UI receives events as each agent completes."""
    if not req.raw_input.strip():
        raise HTTPException(status_code=400, detail="raw_input cannot be empty")

    task_id = req.task_id or f"PROC-{uuid.uuid4().hex[:8].upper()}"

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event_type: str, data: dict):
            queue.put_nowait({"type": event_type, "data": data})

        import concurrent.futures
        loop = asyncio.get_event_loop()

        def run_sync():
            try:
                run_pipeline_streaming(req.raw_input, task_id, emit)
            except Exception as e:
                queue.put_nowait({"type": "error", "data": {"message": str(e)}})
            finally:
                queue.put_nowait(None)

        loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), run_sync)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/run")
def run(req: ProcurementRequest):
    """Non-streaming endpoint (for testing)."""
    if not req.raw_input.strip():
        raise HTTPException(status_code=400, detail="raw_input cannot be empty")
    try:
        result = run_pipeline_streaming(req.raw_input, req.task_id, emit=None)
        packet = result["approval_packet"]
        emo    = result["emo"]
        return {
            "task_id":              packet["task_id"],
            "final_recommendation": packet["final_recommendation"],
            "chain_confidence":     packet["chain_confidence"],
            "was_overturned":       packet["was_overturned"],
            "risk_findings":        packet["risk_findings"],
            "decision_archaeology": packet["decision_archaeology"],
            "hallucination_flags":  emo.all_hallucination_flags(),
            "emo_version":          emo.emo_version,
            "revision_diff":        emo.revision_diff(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
