#!/usr/bin/env python3
"""
WhisperX Transcription Service — UBIK Somatic Node

FastAPI server that exposes audio transcription via the WhisperX library
(GPU-accelerated on RTX 5090) with a fallback to standard Whisper.

Endpoints:
    GET  /health      — liveness + model_loaded status
    POST /transcribe  — accept audio file, return transcript JSON

Usage:
    python whisperx_server.py          # default port 9100
    WHISPERX_PORT=9200 python whisperx_server.py

Author: UBIK Project
Version: 0.1.0
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ubik.whisperx")

app = FastAPI(title="WhisperX Transcription Service", version="0.1.0")

# ---------------------------------------------------------------------------
# Model state (lazy-loaded on first request)
# ---------------------------------------------------------------------------
_model = None          # ("whisperx"|"whisper", model_object)
_model_lock = asyncio.Lock()
_DEVICE = "cuda"
_COMPUTE_TYPE = "float16"   # RTX 5090 / Blackwell — float16 is optimal
_MODEL_NAME = os.getenv("WHISPERX_MODEL", "large-v2")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Return server liveness and model loading status."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": _DEVICE,
        "model_name": _MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = None,
    task: str = "transcribe",
) -> JSONResponse:
    """Transcribe an uploaded audio file.

    Args:
        audio:    Audio file (.mp3, .wav, .m4a, .ogg, .flac).
        language: ISO-639 language code, or None for auto-detect.
        task:     "transcribe" (default) or "translate" (to English).

    Returns:
        JSON with keys: text, language, duration, confidence, segments,
        model_type.
    """
    model_tuple = await _get_model()

    suffix = Path(audio.filename).suffix if audio.filename else ".wav"
    data = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, _do_transcribe, model_tuple, tmp_path, language, task
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_model():
    """Lazy-load model on first request; thread-safe via asyncio lock."""
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                logger.info("Loading transcription model (first request) ...")
                loop = asyncio.get_event_loop()
                _model = await loop.run_in_executor(None, _load_model)
                logger.info("Model loaded: %s", _model[0])
    return _model


def _load_model():
    """Load WhisperX or Whisper model (blocking — runs in executor)."""
    try:
        import whisperx
        model = whisperx.load_model(
            _MODEL_NAME, _DEVICE, compute_type=_COMPUTE_TYPE
        )
        logger.info("Loaded whisperx model '%s' on %s", _MODEL_NAME, _DEVICE)
        return ("whisperx", model)
    except Exception as exc:
        logger.warning("WhisperX not available (%s), falling back to Whisper", exc)
        import whisper
        fallback_name = os.getenv("WHISPER_FALLBACK_MODEL", "base")
        model = whisper.load_model(fallback_name, device=_DEVICE)
        logger.info("Loaded whisper model '%s' on %s", fallback_name, _DEVICE)
        return ("whisper", model)


def _do_transcribe(model_tuple, path: str, language, task: str) -> dict:
    """Run transcription synchronously (called from executor)."""
    kind, model = model_tuple

    if kind == "whisperx":
        import whisperx
        audio = whisperx.load_audio(path)
        kwargs = {"task": task}
        if language:
            kwargs["language"] = language
        result = model.transcribe(audio, **kwargs)
    else:
        import whisper
        kwargs = {"verbose": False, "task": task}
        if language:
            kwargs["language"] = language
        result = model.transcribe(path, **kwargs)

    raw_segments = result.get("segments", [])
    segments = [
        {
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", "").strip(),
        }
        for seg in raw_segments
    ]

    duration = segments[-1]["end"] if segments else None

    confidences = [
        1.0 - seg.get("no_speech_prob", 0.0)
        for seg in raw_segments
        if "no_speech_prob" in seg
    ]
    avg_confidence = sum(confidences) / len(confidences) if confidences else None

    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language", language or "en"),
        "duration": duration,
        "confidence": avg_confidence,
        "segments": segments,
        "model_type": kind,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("WHISPERX_PORT", "9100"))
    logger.info("Starting WhisperX server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
