"""Qwen3-ASR server.

Run this to serve the Qwen3-ASR model as an HTTP endpoint.
Only needed for C#/Rust clients — the Python app loads this in-process.

Usage:
    pip install -r requirements-servers.txt
    python serve_qwen3asr.py
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-ASR Server")

_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading Qwen3-ASR model... (first time download ~4.7GB)")
        try:
            import qwen_asr

            _model = qwen_asr.load("Qwen/Qwen3-ASR-1.7B")
            logger.info("Qwen3-ASR model loaded successfully")
        except ImportError:
            logger.error(
                "qwen-asr is not installed. Install with:\n" "  pip install qwen-asr"
            )
            sys.exit(1)
    return _model


@app.get("/health")
async def health():
    return {"status": "ok", "model": "qwen3-asr-1.7b"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    model = _get_model()
    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        import io

        result = model.transcribe(io.BytesIO(audio_bytes))
        if isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)
        return JSONResponse({"text": text.strip()})
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    logger.info("Starting Qwen3-ASR server on http://localhost:8003")
    logger.info("Model will download on first request if not cached.")
    uvicorn.run(app, host="0.0.0.0", port=8003)
