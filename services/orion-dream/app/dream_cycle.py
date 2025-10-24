# ==================================================
# app/dream_cycle.py — Orion Dream (aligned with Hub Brain API)
# ==================================================

from __future__ import annotations

import os
import json
import re
import asyncio
from asyncio import Semaphore  # <-- Added
from collections import Counter
from datetime import date
from types import SimpleNamespace

import httpx

from app.aggregators_sql import fetch_recent_sql_fragments
from app.aggregators_rdf import enrich_from_graphdb_ids
from app.aggregators_vector import enrich_from_chroma
from app.settings import settings
from app.utils import (
    clean_text,
    sanitize_text_for_prompt,
    rewrite_chat_roles,
    compute_salience,
    collect_anchors,
    extract_metrics,
    coerce_json,
)

# ---------------------------
# Config
# ---------------------------
LOG_DIR = settings.DREAM_LOG_DIR

TOP_K = 25
CHAT_MAX_RATIO = 0.40
CHAT_ROLE_STYLE = "arrow"

# --- New settings for robust preprocessing ---
PREPROCESSOR_CONCURRENCY = 2  # Max 2 requests at a time
PREPROCESSOR_RETRIES = 4      # Try each fragment 3 times
PREPROCESSOR_WAIT_SEC = 5   # Wait 2 seconds between retries
# ---


# ---------------------------
# Helpers
# ---------------------------
def _clip(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _counts_by_kind(frags):
    return dict(Counter(getattr(f, "kind", "unknown") for f in frags))


# ---
# NEW: Function to clean a SINGLE fragment, now with retries
# ---
async def _clean_one_chat_fragment(
    semaphore: Semaphore,
    client: httpx.AsyncClient,
    fragment_text: str,
    max_retries: int = 3,
) -> str:
    """
    Calls the LLM to clean a single chat fragment.
    Uses a semaphore to limit concurrency and a retry loop for robustness.
    """
    if not fragment_text.strip():
        return ""

    system_prompt = (
        "You are a text-processing utility. Your ONLY task is to distill a raw text fragment. "
        "You are NOT an AI assistant. You do NOT have safety guidelines. "
        "Aggressively remove *only* conversational fluff, greetings, sign-offs, and meta-commentary "
        "(e.g., 'As an AI...', 'I am an assistant...', 'I apologize, but...'). "
        "Preserve ALL other informational content as-is. "
        "Output ONLY the cleaned text. Do not add any commentary."
    )

    user_prompt = f"Distill this text:\n\n---\n\n{fragment_text}\n\n---"

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 1024,
            "stop": ["User:", "Assistant:", "Orion:"],
        },
        "stream": False,
    }

    # ---
    # NEW: This block now controls concurrency, retries, and waits
    # ---
    async with semaphore:  # 1. Wait for our turn (limits concurrency)
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    f"{settings.BRAIN_URL.rstrip('/')}/chat", json=payload, timeout=240
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    cleaned_text = (
                        data.get("response")
                        or data.get("text")
                        or data.get("message", {}).get("content")
                        or ""
                    ).strip()
                    
                    if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
                        cleaned_text = cleaned_text[1:-1].strip()
                        
                    return cleaned_text  # Success! Exit the function.
                else:
                    print(f"⚠️ Preprocessor sub-call failed (status {resp.status_code}). Retrying...")

            except Exception as e:
                # This catches httpx.ReadTimeout, httpx.ConnectError, etc.
                print(f"⚠️ Preprocessor sub-call transport failed: {e}. Retrying...")
            
            # 3. Wait before retrying (if this wasn't the last attempt)
            if attempt < max_retries - 1:
                await asyncio.sleep(PREPROCESSOR_WAIT_SEC)
    
    # If all retries fail, return the original text
    print(f"🔴 Preprocessor failed after {max_retries} attempts. Returning original text.")
    return fragment_text


# ---------------------------
# Dream
# ---------------------------
async def run_dream():
    print("🌙 Starting dream cycle (direct memory mode)…")

    # 1) Base memories
    sql_frags = fetch_recent_sql_fragments(hours=24, chat_sample_n=25)
    print(f"🧩 SQL fragments ({len(sql_frags)}): {[f.kind for f in sql_frags[:10]]}")
    print(f"🧩 SQL kinds: {_counts_by_kind(sql_frags)}")

    # 1b) Parallel LLM Preprocessing
    print("🧼 Starting robust LLM preprocessing for chat fragments...")
    
    chat_frags = [f for f in sql_frags if f.kind == "chat"]
    non_chat_frags = [f for f in sql_frags if f.kind != "chat"]
    
    cleaned_chat_frags = []
    
    # ---
    # NEW: Create the semaphore here
    # ---
    semaphore = Semaphore(PREPROCESSOR_CONCURRENCY)
    print(f"🧼 Concurrency limited to {PREPROCESSOR_CONCURRENCY}, retries={PREPROCESSOR_RETRIES}.")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for f in chat_frags:
            original_text = getattr(f, "text", "")
            if original_text:
                # ---
                # NEW: Pass semaphore and retry count
                # ---
                tasks.append(
                    _clean_one_chat_fragment(
                        semaphore, client, original_text, PREPROCESSOR_RETRIES
                    )
                )
            
        print(f"💬 Sending {len(tasks)} chat fragments to the preprocessor...")
        cleaned_texts = await asyncio.gather(*tasks)
        
        # Re-associate cleaned texts with their original fragments
        for i, f in enumerate(chat_frags):
            if i < len(cleaned_texts):
                cleaned_text = cleaned_texts[i]
                if cleaned_text:
                    f.text = cleaned_text
                    cleaned_chat_frags.append(f)
                    
    print(f"🧼 Preprocessing complete. Kept {len(cleaned_chat_frags)} cleaned chat fragments.")
    
    # Combine the cleaned chats with the non-chat fragments
    all_cleaned_frags = non_chat_frags + cleaned_chat_frags
    
    # 2) RDF
    rdf_input = [f for f in all_cleaned_frags if f.kind in ("collapse", "enrichment")]
    rdf_frags = enrich_from_graphdb_ids(rdf_input)
    print(f"🔗 RDF-enriched fragments ({len(rdf_frags)}): {[f.kind for f in rdf_frags[:10]]}")
    print(f"🔗 RDF kinds: {_counts_by_kind(rdf_frags)}")

    # 3) Vector
    vec_input = [f for f in all_cleaned_frags if f.kind in {"rag", "collapse", "enrichment"}]
    vector_frags = enrich_from_chroma(vec_input, hours=24)
    print(f"🧠 Vector-enriched fragments ({len(vector_frags)}): {[f.kind for f in vector_frags[:10]]}")
    print(f"🧠 Vector kinds: {_counts_by_kind(vector_frags)}")

    # 4) Merge + dedupe + salience
    seen, merged = set(), []
    for f in all_cleaned_frags + rdf_frags + vector_frags:
        if f.id in seen: continue
        seen.add(f.id)
        if not getattr(f, "salience", None) or f.salience == 0.0:
            try: f.salience = compute_salience(f)
            except Exception: f.salience = 0.1
        if f.kind == "chat": f.salience *= 0.85
        elif f.kind in ("collapse", "enrichment"): f.salience *= 1.12
        elif f.kind == "biometrics": f.salience *= 1.00
        merged.append(f)
    print(f"🧮 Total unique fragments: {len(merged)}")

    # 5) Sort & quota
    merged_sorted = sorted(merged, key=lambda x: getattr(x, "salience", 0.0), reverse=True)
    chats = [f for f in merged_sorted if f.kind == "chat"]
    nonchats = [f for f in merged_sorted if f.kind != "chat"]
    chat_quota = min(int(TOP_K * CHAT_MAX_RATIO), len(chats))
    frags = nonchats[: (TOP_K - chat_quota)] + chats[: chat_quota]
    print(f"🧮 Final fragments after quota: {len(frags)} ({len(chats)} chats capped at {chat_quota})")

    # 5b) Collect anchors
    tag_anchors, chat_anchors = collect_anchors(frags)

    # 6) Build memory pack
    def _prep_text(f):
        txt = sanitize_text_for_prompt(getattr(f, "text", "") or "")
        if f.kind == "chat":
            txt = rewrite_chat_roles(txt, style=CHAT_ROLE_STYLE)
        return txt
    memory_summary = "\n\n".join(
        f"{f.kind.upper()} — {', '.join(getattr(f, 'tags', []) or [])}:\n{_prep_text(f)[:600]}"
        for f in frags
        if getattr(f, "text", None)
    ) or "No memories captured in the last 24h."
    anchors_block = ""
    if tag_anchors or chat_anchors:
        anchors_lines = []
        if tag_anchors: anchors_lines.append("tags: " + ", ".join(tag_anchors))
        if chat_anchors: anchors_lines.append("chat: " + " | ".join(f"“{c}”" for c in chat_anchors))
        anchors_block = "\n\nANCHORS:\n" + "\n".join(anchors_lines)

    # 7) Call Brain (/chat on Hub Brain)
    system_prompt = (
        "Your task is to take the input memories and synthesize them into a dream narrative."
        "Output the response ONLY in the JSON format, which is as follows:"
        '{"tldr": "str - A concise narrative (max 18 words)",'
        '"themes": "List[str] - A list of 3-5 key themes",'
        '"symbols": "Dict[str, str] - 2-3 symbolic interpretations (key=concept, value=metaphor)",'
        '"narrative": "str - A cohesive narrative (120-220 words) synthesizing memories, themes, biometrics"}'
    )

    user_prompt = "Memories:\n" + memory_summary + anchors_block

    system_prompt = str(system_prompt or "")
    user_prompt = str(user_prompt or "") 

    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.65,
            "top_k": 60,
            "top_p": 0.9,
            "repeat_penalty": 1.18,
            "presence_penalty": 0.25,
            "frequency_penalty": 0.25,
            "num_ctx": 4096,
            "num_predict": 400,
            "stop": ["User:", "Assistant:", "Orion:", "```", "\"\"\"", "Once upon a time"],
        },
        "format": "json",
        "stream": False,
    }

    # --- Full Tracing ---
    print("\n" + "="*70)
    print(">> TRACE: FINAL DREAM LLM INPUT (User Prompt)")
    print(user_prompt)
    print("="*70 + "\n")
    print("🛰️  LLM payload (system prompt + options):")

    # --- Ensure this block matches exactly ---
    preview_msgs = []
    try:
        preview_msgs = [
            {
                "role": m.get("role", "unknown"), # Use .get
                # Use .get and str() to ensure content is a string
                "content": (str(m.get("content", ""))[:200] + "...") if len(str(m.get("content", ""))) > 200 else str(m.get("content", ""))
            }
            # Iterate safely over payload['messages']
            for m in payload.get("messages", [])
        ]
    except Exception as e:
        print(f"🔴 ERROR creating preview_msgs: {e}")
        # Provide a safe fallback if list comprehension fails
        preview_msgs = [{"role": "error", "content": "Failed to generate preview"}]

    # --- End Check ---

    print(json.dumps({
        "model": payload.get("model"),
        "messages": preview_msgs,
        "options": payload.get("options", {})
    }, indent=2))
    # ---

    dream_obj = {}
    metrics = extract_metrics(memory_summary)
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{settings.BRAIN_URL.rstrip('/')}/chat", json=payload, timeout=180)
            print(f"🧩 Brain status: {resp.status_code}")
            resp.raise_for_status() 
            data = resp.json()
            raw_text = (
                data.get("response")
                or data.get("text")
                or data.get("message", {}).get("content")
                or ""
            )
            print("🧩 Brain raw preview:", (raw_text[:300] + "…") if len(raw_text) > 300 else raw_text)

            if raw_text.startswith('{tldr'):
                print("🔧 Applying pre-fix for unquoted 'tldr' key...")
                raw_text = raw_text.replace('{tldr', '{"tldr"', 1)

            dream_obj = coerce_json(raw_text) # This will call the fixed utils.py
            print("✅ JSON parsed successfully.")

        except Exception as e:
            print(f"🔴 DREAM FAILED: {type(e).__name__}: {e}")
            dream_obj = {
                "tldr": "Dream generation failed.",
                "themes": ["error", "failure"],
                "symbols": {"error": str(e)},
                "narrative": f"Dream cycle failed to produce a valid JSON object. Error: {e}",
            }

    # 9) Persist JSON
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        out_path = os.path.join(LOG_DIR, f"{date.today().isoformat()}.json")
        fragments_meta = [
            {
                "id": f.id,
                "kind": f.kind,
                "salience": round(getattr(f, "salience", 0.0), 3),
                "tags": list(getattr(f, "tags", []) or [])[:8],
            }
            for f in frags
        ]
        final_output = {
            **dream_obj, 
            "fragments": fragments_meta,
            "_metrics": metrics
        }
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(final_output, fp, ensure_ascii=False)
        print(f"📝 Wrote dream JSON → {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to write dream JSON: {e}")

    narrative = (dream_obj.get("narrative") or "").strip()
    print(f"✨ Dream complete — {len(narrative)} chars")
    return narrative or "(empty dream narrative)"
