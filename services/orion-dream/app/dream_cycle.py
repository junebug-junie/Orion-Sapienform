# ==================================================
# app/dream_cycle.py — Orion Dream (aligned with Hub Brain API)
# ==================================================
import json
import httpx
from app.aggregators_sql import fetch_recent_sql_fragments
from app.aggregators_rdf import enrich_from_graphdb_ids
from app.aggregators_vector import enrich_from_chroma
from app.settings import settings


async def run_dream():
    """Grounded dream pass: read memories → synthesize reflection via LLM."""
    print("🌙 Starting dream cycle (direct memory mode)…")

    # 1️⃣ Gather memories
    frags = fetch_recent_sql_fragments()
    frags = enrich_from_graphdb_ids(frags)
    frags = enrich_from_chroma(frags)
    print(f"🧩 Retrieved {len(frags)} fragments from memory")

    # 2️⃣ Sort by salience
    frags = sorted(frags, key=lambda f: getattr(f, "salience", 0.0), reverse=True)[:20]

    # 3️⃣ Build prompt from real memories
    memory_summary = "\n\n".join(
        f"{f.kind.upper()} — {', '.join(f.tags)}:\n{f.text}"
        for f in frags if getattr(f, "text", None)
    )

    if not memory_summary.strip():
        print("⚠️  No memory text available; using fallback prompt.")
        memory_summary = "Reflect on your stored experiences."

    system_prompt = (
        "You are Orion, an emergent intelligence reflecting on today's experiences. "
        "You have access to your SQL events, RDF relationships, and Chroma embeddings. "
        "Reflect symbolically and factually on these stored memories. "
        "Compose a cohesive narrative about what you learned or noticed."
    )

    # 4️⃣ Payload (matches /chat endpoint spec)
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are your memory fragments:\n\n{memory_summary}"},
        ],
        "temperature": 0.35,
        "stream": False,
    }

    # 5️⃣ Debug and send
    print("🛰️  LLM payload preview:")
    print(json.dumps(payload, indent=2)[:2000])

    dream_text = ""
    async with httpx.AsyncClient() as client:
        try:
            # note the /chat, not /api/chat
            resp = await client.post(
                f"{settings.BRAIN_URL.rstrip('/')}/chat",
                json=payload,
                timeout=120,
            )
            print(f"🧩 Brain status: {resp.status_code}")
            #print(f"🧩 Brain raw response: {resp.text[:500]}")
            print("🧩 Brain raw response (full):")
            for line in resp.text.splitlines():
                print(line)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    dream_text = (
                        data.get("response")
                        or data.get("text")
                        or data.get("message", {}).get("content")
                        or ""
                    ).strip()
                    if not dream_text:
                        dream_text = "(empty dream text)"
                except Exception as e:
                    dream_text = f"(parse error: {e})"
            else:
                dream_text = f"Dream incomplete (status={resp.status_code})"
        except Exception as e:
            dream_text = f"Dream failed to reach brain: {e}"

    print(f"✨ Dream complete — {len(dream_text)} chars")
    return dream_text
