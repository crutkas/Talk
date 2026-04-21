"""Canary Qwen 2.5B transcription server.

Run this to serve the Canary Qwen STT model as an HTTP endpoint
for the C#/Rust clients or when you don't want to load the model in-process.

Usage:
    pip install -r requirements-servers.txt
    python serve_canary.py

The model will be downloaded on first run (~5GB).
"""

from __future__ import annotations

import io
import logging
import sys
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Canary Qwen 2.5B STT Server")

# Lazy-loaded model
_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("Loading Canary Qwen 2.5B model... (first time download ~5GB)")
        try:
            from nemo.collections.asr.models import ASRModel

            _model = ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
            logger.info("Canary Qwen model loaded successfully")
        except ImportError:
            logger.error(
                "nemo_toolkit is not installed. Install with:\n"
                "  pip install nemo_toolkit[asr]"
            )
            sys.exit(1)
    return _model


@app.get("/health")
async def health():
    return {"status": "ok", "model": "canary-qwen-2.5b"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    model = _get_model()
    audio_bytes = await audio.read()

    # Write to temp file (NeMo expects file paths)
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        result = model.transcribe([temp_path])
        text = result[0] if isinstance(result, list) else str(result)
        return JSONResponse({"text": text.strip()})
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    logger.info("Starting Canary Qwen server on http://localhost:8001")
    logger.info("Model will download on first request if not cached.")
    uvicorn.run(app, host="0.0.0.0", port=8001)
