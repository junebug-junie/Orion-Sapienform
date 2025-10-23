# ==================================================
# app/dream_cycle.py — Orion Dream (aligned with Hub Brain API)
# ==================================================
import json
import httpx
from collections import Counter

from app.aggregators_sql import fetch_recent_sql_fragments
from app.aggregators_rdf import enrich_from_graphdb_ids
from app.aggregators_vector import enrich_from_chroma
from app.settings import settings


def _counts_by_kind(frags):
    return dict(Counter(getattr(f, "kind", "unknown") for f in frags))


async def run_dream():
    print("🌙 Starting dream cycle (direct memory mode)…")

    # --- SQL fragments (raw memories)
    sql_frags = fetch_recent_sql_fragments(hours=24, chat_sample_n=15)
    print(f"🧩 SQL fragments: total={len(sql_frags)} kinds={_counts_by_kind(sql_frags)}")

    # --- RDF enrichment (collapse + enrichment only)
    rdf_input = [f for f in sql_frags if f.kind in ("collapse", "enrichment")]
    rdf_frags = enrich_from_graphdb_ids(rdf_input)
    print(f"🔗 RDF-enriched fragments: total={len(rdf_frags)} kinds={_counts_by_kind(rdf_frags)}")

    # --- Vector enrichment (collapse + chat)
    vec_input = [f for f in sql_frags if f.kind in {"rag", "collapse", "enrichment"}]
    vector_frags = enrich_from_chroma(vec_input, hours=24)
    print(f"🧠 Vector-enriched fragments: total={len(vector_frags)} kinds={_counts_by_kind(vector_frags)}")

    # --- Merge unique fragments
    seen = set()
    all_frags = []
    for f in sql_frags + rdf_frags + vector_frags:
        if f.id not in seen:
            seen.add(f.id)
            all_frags.append(f)
    print(f"🧮 Total unique fragments: {len(all_frags)}")

    # ✅ assign unified fragments list
    frags = all_frags

    # --- Sort by salience and cap
    frags = sorted(frags, key=lambda f: getattr(f, "salience", 0.0), reverse=True)[:25]

    # --- Build memory summary
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

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are your memory fragments:\n\n{memory_summary}"},
        ],
        "temperature": 0.35,
        "stream": False,
    }

    print("🛰️  LLM payload preview:")
    print(json.dumps(payload, indent=2)[:2000])

    dream_text = ""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{settings.BRAIN_URL.rstrip('/')}/chat",
                json=payload,
                timeout=120,
            )
            print(f"🧩 Brain status: {resp.status_code}")
            print("🧩 Brain raw response (full):")
            for line in resp.text.splitlines():
                print(line)

            if resp.status_code == 200:
                data = resp.json()
                dream_text = (
                    data.get("response")
                    or data.get("text")
                    or data.get("message", {}).get("content")
                    or ""
                ).strip()
                if not dream_text:
                    dream_text = "(empty dream text)"
            else:
                dream_text = f"Dream incomplete (status={resp.status_code})"
        except Exception as e:
            dream_text = f"Dream failed to reach brain: {e}"

    print(f"✨ Dream complete — {len(dream_text)} chars")
    return dream_text
