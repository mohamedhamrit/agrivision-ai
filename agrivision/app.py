"""REST web endpoint for AgriVision AI.

Provides a working web endpoint for the OpenCV 5 crop-disease pipeline so the
competition submission can be demonstrated live over HTTP.

Run:
    python -m uvicorn agrivision.app:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /            -> service metadata
GET  /health      -> health check (model + OpenCV versions)
POST /analyze     -> raw video bytes in the body (no multipart dependency);
                     optional ?filename=demo.mp4 query to hint the container.
POST /analyze_url -> JSON {"video_url": "http(s)://..."}; downloads then analyzes.
GET  /examples    -> runs the four built-in synthetic clips and returns the report.

Response shape (a single record):
    {source, analysis: {...}, action: {...}, store_key}
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request

import cv2  # noqa: F401  (version reported by /health)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent import Agent
from .aws import Notifier, PipelineLambda, Store
from .pipeline import OpenCVPipeline

_LEVELS = ("healthy", "mild", "moderate", "severe")
_RATIOS = {"healthy": 0.0, "mild": 0.07, "moderate": 0.40, "severe": 0.70}

app = FastAPI(
    title="AgriVision AI",
    version="0.1.0",
    description="OpenCV 5 plant-disease detection and agentic crop monitoring endpoint.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the shared pipeline/agent once at startup (loads leaf_cnn.h5).
_pipeline = OpenCVPipeline()
_agent = Agent()
_store = Store(local_mode=True)
_notifier = Notifier(local_mode=True)
_handler = PipelineLambda(_pipeline, _agent, _store, _notifier)


def _analyze_bytes(data: bytes, filename: str = "upload.mp4") -> dict:
    suffix = os.path.splitext(filename)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix="agrivision_", suffix=suffix)
    os.close(fd)  # mkstemp pre-creates the file; release the handle
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        payload = _handler.handler({"objects": [tmp_path]})["body"]
        records = json.loads(payload) if isinstance(payload, str) else payload
        return records[0]
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/")
def root() -> dict:
    return {
        "service": "AgriVision AI",
        "description": "OpenCV 5 + Transfer-Learning CNN crop-disease monitoring",
        "endpoints": ["/health", "/analyze", "/analyze_url", "/examples"],
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "opencv": cv2.__version__,
        "model": "agrivision/models/leaf_cnn.h5 (MobileNetV2 transfer, 128x128)",
    }


@app.post("/analyze")
async def analyze(request: Request, filename: str = "upload.mp4") -> JSONResponse:
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Empty body: POST raw video bytes")
    try:
        record = _analyze_bytes(data, filename)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}") from exc
    return JSONResponse(content=record)


@app.post("/analyze_url")
async def analyze_url(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Expected a JSON body") from exc
    url = (payload or {}).get("video_url")
    if not url:
        raise HTTPException(status_code=400, detail='JSON body must contain "video_url"')
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Only http(s) URLs are supported in local mode")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        filename = os.path.basename(url) or "remote.mp4"
        record = _analyze_bytes(data, filename)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=422, detail=f"Download/analysis failed: {exc}") from exc
    return JSONResponse(content=record)


@app.get("/examples")
def examples() -> JSONResponse:
    from .sample_data import get_default_videos_dir, make_video

    videos_dir = get_default_videos_dir()
    clips = []
    for level in _LEVELS:
        path = os.path.join(videos_dir, f"{level}.mp4")
        if not os.path.exists(path):
            make_video(_RATIOS[level], path, seed=_LEVELS.index(level))
        clips.append(path)
    payload = _handler.handler({"objects": clips})["body"]
    records = json.loads(payload) if isinstance(payload, str) else payload
    return JSONResponse(content=records)