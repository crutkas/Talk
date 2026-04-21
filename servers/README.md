# Server Setup Guide

## Overview

The Python prototype runs STT models **in-process** — no servers needed for basic use.

Servers are needed when:
- You're using the **C# or Rust** clients (they can't load Python models directly)
- You want to **offload models to a GPU machine** and run the client on a thin laptop
- You want to keep models **persistent in memory** across app restarts

## Quick Start

```bash
cd servers
pip install -r requirements-servers.txt
```

## Which servers do I need?

| Engine | Python app | C#/Rust app | Server script |
|--------|-----------|-------------|---------------|
| Whisper | In-process ✅ | In-process (whisper.cpp) ✅ | Not needed |
| Canary Qwen | HTTP server required | HTTP server required | `serve_canary.py` |
| Voxtral | HTTP server required | HTTP server required | `serve_voxtral.py` |
| Qwen3-ASR | In-process ✅ | HTTP server required | `serve_qwen3asr.py` |
| Translation | In-process ✅ | HTTP server required | `serve_translation.py` |

## Starting servers

### Canary Qwen 2.5B (port 8001)
```bash
# Requires: pip install nemo_toolkit[asr]
python serve_canary.py
# Model downloads on first request (~5GB)
```

### Voxtral Transcribe 2 (port 8002)
```bash
# Requires: pip install transformers torch torchaudio
python serve_voxtral.py
# Model downloads on first request (~8.9GB)
```

### Qwen3-ASR (port 8003)
```bash
# Requires: pip install qwen-asr
python serve_qwen3asr.py
# Model downloads on first request (~4.7GB)
```

### Translation (port 8010)
```bash
# Choose one:
python serve_translation.py --model nllb-200      # 200+ languages, ~2.5GB
python serve_translation.py --model seamless-m4t   # 100+ languages, ~4.5GB
python serve_translation.py --model madlad-400     # 400+ languages, ~6GB
```

## Model download sizes

| Model | Download Size | VRAM/RAM needed |
|-------|--------------|-----------------|
| Whisper large-v3-turbo | ~3GB | ~4GB |
| Canary Qwen 2.5B | ~5GB | ~6GB |
| Voxtral Mini 4B | ~8.9GB | ~10GB |
| Qwen3-ASR 1.7B | ~4.7GB | ~5GB |
| NLLB-200 1.3B | ~2.5GB | ~3GB |
| SeamlessM4T v2 | ~4.5GB | ~6GB |
| Madlad-400 3B | ~6GB | ~7GB |

All models are cached in `~/.cache/huggingface/hub/` after first download.

## Health checks

Each server exposes a `/health` endpoint:
```bash
curl http://localhost:8001/health   # Canary Qwen
curl http://localhost:8002/health   # Voxtral
curl http://localhost:8003/health   # Qwen3-ASR
curl http://localhost:8010/health   # Translation
```

## Running on a remote GPU machine

1. Start the server(s) on the GPU machine
2. Update `config.json` endpoints to point to the remote IP:
```json
{
  "models": {
    "canary_qwen": {
      "endpoint": "http://gpu-machine:8001/transcribe"
    }
  }
}
```
