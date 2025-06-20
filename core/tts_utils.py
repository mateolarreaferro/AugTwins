"""
Thin wrapper around ElevenLabs TTS.

Keeps all voice-specific networking in one place.
"""
from __future__ import annotations
import os
import requests
from uuid import uuid4
from typing import Optional

# key import or env fallback 
try:
    from settings import ELEVEN_API_KEY as _ELEVEN_KEY
except (ModuleNotFoundError, ImportError):
    try:
        from settings import ELEVENLABS_API_KEY as _ELEVEN_KEY
    except (ModuleNotFoundError, ImportError):
        _ELEVEN_KEY = os.getenv("ELEVEN_API_KEY", "") or os.getenv("ELEVENLABS_API_KEY", "")


# main helper
def speak(text: str, voice_id: str, *, playback_cmd: str = "afplay") -> None:
    """
    Download TTS audio from ElevenLabs and play it via *playback_cmd*.
    No-ops if keys or voice_id are missing.
    """
    if not _ELEVEN_KEY or not voice_id:
        print("[TTS disabled – set ELEVEN_API_KEY / ELEVENLABS_API_KEY and voice ID]")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": _ELEVEN_KEY, "Content-Type": "application/json"}
    body = {"text": text,
            "voice_settings": {"stability": 0.55, "similarity_boost": 0.8}}

    r = requests.post(url, json=body, headers=headers, timeout=30)
    if r.status_code != 200:
        print("[TTS error]", r.text)
        return

    fname = f"audio_{uuid4()}.mp3"
    with open(fname, "wb") as f:
        f.write(r.content)
    os.system(f"{playback_cmd} {fname}")
    os.remove(fname)
