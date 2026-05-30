# PR: fix(hub) — voice STT capture and silent-audio handling

**Branch:** `fix/hub-voice-tts-hang`  
**Base:** `main`

## Summary

Fixes Hub hold-to-talk returning empty transcripts when the browser sent valid-sized audio that was effectively silent (WebM/MediaRecorder path) or when Whisper received near-zero PCM levels.

- **Hub client:** Web Audio PCM capture → 16 kHz mono WAV (replaces MediaRecorder/WebM for STT upload); peak-amplitude gate; pointer events; stop-before-mic-ready UX.
- **Hub WS:** `audio_bytes` logging; differentiated errors for silent mic vs no speech; surfaces STT bus `system.error` details.
- **whisper-tts:** ffmpeg WebM/OGG path retained; WAV level measurement; near-silent skip; typed `stt.transcribe.result` envelopes with metadata; `system.error` on failure.

## Test plan

- [ ] Hard refresh Hub; hold mic — console shows `[voice] peak amplitude` > 0.01 when speaking.
- [ ] Short utterance (~0.5s) transcribes when mic level is healthy.
- [ ] Muted/wrong input → client or server “microphone level too low” message.
- [ ] `python3 -m pytest services/orion-whisper-tts/tests/test_stt_engine.py` (in whisper image or venv with deps).
