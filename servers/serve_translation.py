"""Translation model server.

Serves NLLB-200, SeamlessM4T v2, or Madlad-400 as an HTTP endpoint.
Only needed for C#/Rust clients — the Python app loads these in-process.

Usage:
    pip install -r requirements-servers.txt
    python serve_translation.py                     # defaults to nllb-200
    python serve_translation.py --model seamless-m4t
    python serve_translation.py --model madlad-400
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Translation Server")

_engine = None


class TranslateRequest(BaseModel):
    text: str
    source_lang: str = "en"
    target_lang: str = "es"


def _get_engine():
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized — this shouldn't happen")
    return _engine


@app.get("/health")
async def health():
    engine = _get_engine()
    return {"status": "ok", "model": engine.name}


@app.post("/translate")
async def translate(req: TranslateRequest):
    engine = _get_engine()
    try:
        result = engine.translate(req.text, req.source_lang, req.target_lang)
        return JSONResponse({"text": result})
    except Exception as e:
        logger.error("Translation failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/languages")
async def languages():
    engine = _get_engine()
    return {"languages": engine.supported_languages()}


def main():
    global _engine

    parser = argparse.ArgumentParser(description="Translation model server")
    parser.add_argument(
        "--model",
        choices=["nllb-200", "seamless-m4t", "madlad-400"],
        default="nllb-200",
        help="Translation model to serve (default: nllb-200)",
    )
    parser.add_argument("--port", type=int, default=8010, help="Port (default: 8010)")
    parser.add_argument("--device", default="auto", help="Device (default: auto)")
    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "python"))
    from src.translation.factory import create_translation_engine

    logger.info("Loading %s translation model...", args.model)
    _engine = create_translation_engine(args.model, {"device": args.device})

    if _engine.needs_download():
        logger.info("Model will be downloaded on first use. This is a one-time operation.")

    # Force model load now so the server is ready
    _engine.download_model(
        progress_callback=lambda msg: logger.info(msg)
    )

    logger.info("Starting translation server on http://localhost:%d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
