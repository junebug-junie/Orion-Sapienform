# Orion Interface

The **interface/** package contains all human-facing interaction layers for Orion. It is the outward-facing gateway where users connect with Orion's cognition, memory, and perception systems.

---

## Purpose

* **Hub**: The central multimodal gateway (voice, text, future vision input).
* **Dashboards**: Observability tools, dev-facing UIs, and visualization panels.

Together, these modules form Orion’s "front door" — translating human input into events and presenting responses, states, and insights back to users.

---

## Folder Structure

```
interface/
├── hub/          # Main interaction hub (was orion-voice-service)
│   ├── scripts/  # FastAPI app, ASR, TTS, routing
│   ├── static/   # Frontend assets (JS, CSS)
│   ├── templates/# HTML templates for UI
│   └── Dockerfile
│
└── dashboards/   # Observability, debug panels, analytics (future)
```

---

## Responsibilities

* Provide **real-time interaction** (voice chat, text chat).
* Expose a **browser-accessible UI** for interaction.
* **Publish events** (transcripts, LLM responses, TTS) to Orion Bus.
* Serve as the **boundary layer** between human input and Orion’s cognition.

---

## Future Roadmap

* [ ] Add typed input & conversation history UI.
* [ ] Integrate vision events & multimodal state display.
* [ ] Build dashboards for memory visualization & agent state tracing.
* [ ] Add WebSocket API for programmatic clients.

---

## Dev Notes

* The hub uses **FastAPI + WebSocket** for low-latency bi-directional communication.
* STT runs locally via **Faster-Whisper** for privacy & GPU acceleration.
* TTS uses **Coqui TTS** container for speech synthesis.
* UI served via **Caddy** (reverse proxy, dev/prod mode, HTTPS in prod).
