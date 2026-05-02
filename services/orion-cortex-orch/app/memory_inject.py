from __future__ import annotations

import json
import logging

logger = logging.getLogger("orion.cortex.memory_inject")

try:
    import psycopg2  # type: ignore
    import psycopg2.extensions  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None


def _time_horizon_hint(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return ""
    if not isinstance(raw, dict):
        return ""
    kind = str(raw.get("kind") or "")
    if kind == "timeless":
        return "timeless"
    if kind == "era_bound":
        return f"era:{raw.get('start', '')}-{raw.get('end', '')}".strip("-")
    if kind == "current":
        return "current"
    if kind == "expiring":
        return f"expires:{raw.get('end', '')}"
    return kind or ""


def fetch_always_inject_block(*, lane: str, token_budget: int, dsn: str) -> str:
    """
    Blocking fetch of always-inject memory cards for prompt context.
    Uses psycopg2 with a short statement timeout (best-effort).
    """
    if not dsn or not str(dsn).strip():
        return ""
    if psycopg2 is None:
        return ""
    try:
        conn = psycopg2.connect(dsn, connect_timeout=1)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET LOCAL statement_timeout = '250ms'")
        cur.execute(
            """
            SELECT anchor_class, types, summary, time_horizon
            FROM memory_cards
            WHERE priority = 'always_inject'
              AND status = 'active'
              AND (
                visibility_scope @> ARRAY['all']::text[]
                OR %s = ANY(visibility_scope)
              )
            ORDER BY updated_at DESC
            LIMIT 64
            """,
            (lane,),
        )
        rows = cur.fetchall() or []
        cur.execute(
            """
            SELECT subschema
            FROM memory_cards
            WHERE status = 'active'
              AND subschema ? 'memory_graph'
              AND (
                visibility_scope @> ARRAY['all']::text[]
                OR %s = ANY(visibility_scope)
              )
            ORDER BY updated_at DESC
            LIMIT 48
            """,
            (lane,),
        )
        mg_rows = cur.fetchall() or []
        cur.close()
        conn.close()
    except Exception as exc:
        logger.warning("always_inject fetch failed: %s", exc)
        return ""

    bullets: list[str] = []
    est = 0
    for anchor_class, types, summary, th in rows:
        type0 = (types or ["fact"])[0] if types else "fact"
        label = anchor_class or type0
        hint = _time_horizon_hint(th)
        suffix = f", {hint}" if hint else ""
        line = f"- {summary} ({label}{suffix})"
        est += max(len(line.split()), 1) * 1.3
        if est > float(token_budget):
            break
        bullets.append(line)
    for (sub_raw,) in mg_rows:
        sub: dict
        if isinstance(sub_raw, str):
            try:
                sub = json.loads(sub_raw)
            except Exception:
                sub = {}
        elif isinstance(sub_raw, dict):
            sub = sub_raw
        else:
            sub = {}
        mg = sub.get("memory_graph") if isinstance(sub.get("memory_graph"), dict) else {}
        facts = mg.get("facts") if isinstance(mg.get("facts"), list) else []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            subj = str(fact.get("subject") or "").strip()
            pred = str(fact.get("predicate") or "").strip()
            obj = str(fact.get("object") or "").strip()
            pol = str(fact.get("polarity") or "").strip()
            tim = str(fact.get("time") or "").strip()
            bits = [subj, pred, obj]
            line = "- " + " ".join(b for b in bits if b)
            if tim:
                line += f" ({tim})"
            if pol:
                line += f" [{pol}]"
            est += max(len(line.split()), 1) * 1.3
            if est > float(token_budget):
                break
            bullets.append(line)
        if est > float(token_budget):
            break
    if not bullets:
        return ""
    return "[Known facts about Juniper]\n" + "\n".join(bullets) + "\n"
