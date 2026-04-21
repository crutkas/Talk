"""Voxtral Transcribe 2 server.

Run this to serve the Voxtral STT model as an HTTP endpoint.

Usage:
    pip install -r requirements-servers.txt
    python serve_voxtral.py

The model will be downloaded on first run (~8.9GB).
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

app = FastAPI(title="Voxtral Transcribe 2 Server")

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        logger.info("Loading Voxtral Transcribe 2... (first time download ~8.9GB)")
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            model_id = "mistralai/Voxtral-Mini-4B-Realtime-2602"
            _processor = AutoProcessor.from_pretrained(model_id)
            _model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            logger.info("Voxtral model loaded successfully")
        except ImportError:
            logger.error(
                "transformers is not installed. Install with:\n"
                "  pip install transformers torch torchaudio"
            )
            sys.exit(1)
    return _model, _processor


@app.get("/health")
async def health():
    return {"status": "ok", "model": "voxtral-mini-4b-realtime"}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    model, processor = _get_model()
    audio_bytes = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(temp_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(outputs[0], skip_special_tokens=True)
        return JSONResponse({"text": text.strip()})
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    logger.info("Starting Voxtral server on http://localhost:8002")
    logger.info("Model will download on first request if not cached.")
    uvicorn.run(app, host="0.0.0.0", port=8002)
