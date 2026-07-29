"""AgriBloom V2 voice package.

Portable speech I/O, decoupled from any UI so the FastAPI backend and every
channel (web, WhatsApp, Android) can reuse it:

  * STT (:mod:`voice.stt`): faster-whisper (local, GPU) -> Groq Whisper (free API)
  * TTS (:mod:`voice.tts`): edge-tts (free neural Indic voices) -> gTTS

Both degrade gracefully and auto-handle 10+ Indian languages.
"""
from voice.stt import transcribe  # noqa: F401
from voice.tts import synthesize  # noqa: F401
