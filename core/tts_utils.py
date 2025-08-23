"""
Thin wrapper around ElevenLabs TTS.

Keeps all voice-specific networking in one place.
"""
from __future__ import annotations
import os
import asyncio
import json
import base64
import ssl
import websockets
try:
    import requests
except ModuleNotFoundError:  # allow tests without requests
    requests = None
from uuid import uuid4
from typing import Optional, AsyncGenerator, Dict, Any

# API key - load from centralized config
from config import ELEVEN_API_KEY


class ElevenLabsRealtimeSession:
    """Manages a persistent WebSocket connection to ElevenLabs Realtime API."""
    
    def __init__(self, voice_id: str, api_key: str = None):
        self.voice_id = voice_id
        self.api_key = api_key or ELEVEN_API_KEY
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        
    async def connect(self) -> None:
        """Establish WebSocket connection to ElevenLabs Realtime API."""
        if self.is_connected and self.websocket:
            print(f"[TTS Debug] Already connected to voice {self.voice_id}")
            return
            
        url = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream-input?output_format=pcm_22050"
        headers = {"xi-api-key": self.api_key}
        
        print(f"[TTS Debug] Connecting to {url} with voice {self.voice_id}")
        try:
            # Create SSL context for macOS compatibility
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                url, additional_headers=headers, ping_interval=20, ping_timeout=10, ssl=ssl_context
            )
            print(f"[TTS Debug] WebSocket connected successfully")
            await self._configure_session()
            self.is_connected = True
            print(f"[TTS Debug] Session configured and ready")
        except Exception as e:
            print(f"[TTS Debug] Connection failed: {e}")
            raise
        
    async def _configure_session(self) -> None:
        """Configure the session for PCM S16LE output at 22.05 kHz."""
        config = {
            "text": " ",
            "voice_settings": {
                "speed": 1,
                "stability": 0.55,
                "similarity_boost": 0.8
            },
            "xi_api_key": self.api_key
        }
        print(f"[TTS Debug] Sending config: {json.dumps(config, indent=2)}")
        await self.websocket.send(json.dumps(config))
        print(f"[TTS Debug] Config sent (not waiting for response)")
                
    async def stream_text_to_pcm(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream text to ElevenLabs and yield PCM audio chunks."""
        if not self.is_connected:
            print(f"[TTS Debug] Not connected, connecting now...")
            await self.connect()
            
        message = {
            "text": text,
            "try_trigger_generation": True
        }
        print(f"[TTS Debug] Sending text message: {text[:50]}...")
        await self.websocket.send(json.dumps(message))
        
        # Send end-of-input signal
        end_message = {"text": ""}
        print(f"[TTS Debug] Sending end-of-input signal...")
        await self.websocket.send(json.dumps(end_message))
        
        chunk_count = 0
        while True:
            try:
                print(f"[TTS Debug] Waiting for response (chunk {chunk_count})...")
                response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                
                if isinstance(response, bytes) and len(response) > 0:
                    print(f"[TTS Debug] Received audio chunk {chunk_count}: {len(response)} bytes")
                    chunk_count += 1
                    yield response
                elif isinstance(response, str):
                    print(f"[TTS Debug] Received text response: {response}")
                    data = json.loads(response)
                    if "audio" in data and data["audio"]:
                        # Decode base64 audio data
                        audio_bytes = base64.b64decode(data["audio"])
                        print(f"[TTS Debug] Decoded audio chunk {chunk_count}: {len(audio_bytes)} bytes")
                        chunk_count += 1
                        yield audio_bytes
                    if data.get("isFinal"):
                        print(f"[TTS Debug] Received isFinal, ending stream")
                        break
                    if "error" in data:
                        print(f"[TTS Debug] Received error: {data['error']}")
                        break
                else:
                    print(f"[TTS Debug] Received unknown response type: {type(response)}")
                    
            except asyncio.TimeoutError:
                print(f"[TTS Debug] Timeout waiting for response after chunk {chunk_count}")
                break
            except websockets.exceptions.ConnectionClosedOK:
                print(f"[TTS Debug] WebSocket closed normally after {chunk_count} chunks")
                break
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[TTS Debug] WebSocket closed: {e}")
                self.is_connected = False
                break
            except Exception as e:
                print(f"[TTS Stream Error] Unexpected error: {type(e).__name__}: {e}")
                self.is_connected = False
                break
                
        print(f"[TTS Debug] Stream ended, total chunks: {chunk_count}")
                
    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.websocket = None


class RealtimeTTSManager:
    """Manages TTS sessions with cancellation support."""
    
    def __init__(self):
        self.sessions: Dict[str, ElevenLabsRealtimeSession] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
    async def get_or_create_session(self, voice_id: str) -> ElevenLabsRealtimeSession:
        """Get existing session or create new one for voice."""
        if voice_id not in self.sessions:
            self.sessions[voice_id] = ElevenLabsRealtimeSession(voice_id)
            await self.sessions[voice_id].connect()
        return self.sessions[voice_id]
        
    async def stream_tts(self, text: str, voice_id: str, job_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream TTS with metadata for WebSocket transmission."""
        job_id = job_id or str(uuid4())
        
        try:
            session = await self.get_or_create_session(voice_id)
            
            yield {
                "type": "audio_start",
                "id": job_id,
                "encoding": "pcm_s16le",
                "sample_rate": 22050,
                "channels": 1
            }
            
            chunk_count = 0
            async for pcm_chunk in session.stream_text_to_pcm(text):
                yield {
                    "type": "audio_data",
                    "id": job_id,
                    "data": pcm_chunk,
                    "chunk_index": chunk_count
                }
                chunk_count += 1
                
                if job_id in self.active_jobs and self.active_jobs[job_id].cancelled():
                    break
                    
        except Exception as e:
            print(f"[TTS Manager Error] {e}")
        finally:
            yield {"type": "audio_end", "id": job_id}
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
                
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active TTS job."""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].cancel()
            return True
        return False
        
    async def close_all(self) -> None:
        """Close all sessions."""
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()
        for task in self.active_jobs.values():
            task.cancel()
        self.active_jobs.clear()


# Global TTS manager instance
_tts_manager = None

def get_tts_manager() -> RealtimeTTSManager:
    """Get the global TTS manager instance."""
    global _tts_manager
    if _tts_manager is None:
        _tts_manager = RealtimeTTSManager()
    return _tts_manager


# Legacy synchronous helper (backward compatibility)
def speak(text: str, voice_id: str, *, playback_cmd: str = "afplay") -> None:
    """
    Download TTS audio from ElevenLabs and play it via *playback_cmd*.
    No-ops if keys or voice_id are missing.
    """
    if not ELEVEN_API_KEY or not voice_id or not requests:
        print("[TTS disabled – set ELEVEN_API_KEY / ELEVENLABS_API_KEY and voice ID]")
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
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
