from __future__ import annotations

import json
import uuid
import psycopg2
from statistics import mean
from datetime import datetime, timedelta
from typing import List, Optional

from app.utils import Fragment
from app.settings import settings


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 2) if xs else None


def _summarize_biometrics(cur, hours: int, limit_rows: int = 500) -> Optional[dict]:
    """
    Input columns:
      - gpu (JSON): {"latest_file":"...", "gpus":[
            {"timestamp":"...","gpu_index":"0","gpu_name":" ...",
             "utilization_gpu":" 0","memory_used_mb":" 1022",
             "memory_total_mb":" 16380","power_draw_watts":" 7.43"}, ...]}
      - cpu (JSON): {"coretemp-isa-0001": {"Package id 1": {"temp1_input":41.0,...}, "Core 0": {...}, ...}, ...}

    Output:
      {"gpu_power_w": float|None,
       "gpu_util_pct": float|None,
       "gpu_mem_mb": float|None,
       "cpu_temp_c": float|None}
    """
    hours = max(1, min(int(hours), 168))  # guard
    cur.execute(f"""
        SELECT timestamp, gpu, cpu
        FROM orion_biometrics
        WHERE timestamp::timestamptz > NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp::timestamptz ASC
        LIMIT {limit_rows};
    """)
    rows = cur.fetchall()
    if not rows:
        return None

    gpu_power, gpu_util, gpu_mem = [], [], []
    cpu_temps = []

    for ts, gpu_json, cpu_json in rows:
        # --- GPU ---
        try:
            g = gpu_json if isinstance(gpu_json, dict) else json.loads(gpu_json or "{}")
            for item in g.get("gpus", []):
                gpu_power.append(_to_float(item.get("power_draw_watts")))
                gpu_util.append(_to_float(item.get("utilization_gpu")))
                gpu_mem.append(_to_float(item.get("memory_used_mb")))
        except Exception:
            pass

        # --- CPU ---
        try:
            c = cpu_json if isinstance(cpu_json, dict) else json.loads(cpu_json or "{}")
            temps = []
            # payload: top-level sensors (e.g. "coretemp-isa-0000" / "coretemp-isa-0001")
            for sensor in c.values():
                if not isinstance(sensor, dict):
                    continue
                # each sensor has "Package id X", "Core N", etc. → dicts of temp*_input keys
                for vals in sensor.values():
                    if not isinstance(vals, dict):
                        continue
                    for k, v in vals.items():
                        if isinstance(k, str) and k.endswith("_input"):
                            tv = _to_float(v)
                            if tv is not None:
                                temps.append(tv)
            if temps:
                cpu_temps.append(mean(temps))
        except Exception:
            pass

    return {
        "gpu_power_w": _avg(gpu_power),
        "gpu_util_pct": _avg(gpu_util),
        "gpu_mem_mb": _avg(gpu_mem),
        "cpu_temp_c": _avg(cpu_temps),
    }


def fetch_recent_sql_fragments(hours: int = 24, chat_sample_n: int = 200) -> List[Fragment]:
    """
    Returns fragments from:
      - collapse_mirror (timestamp is string → cast to timestamptz)
      - collapse_enrichment (ts is timestamp)
      - chat_history_log (RANDOM sample within 24h, capped at chat_sample_n)
      - orion_biometrics (summarized into a single Fragment)
    """
    hours = max(1, min(int(hours), 168))  # guard
    conn = psycopg2.connect(settings.POSTGRES_URI)
    cur = conn.cursor()
    since_dt = datetime.utcnow() - timedelta(hours=hours)

    # --- Collapse Mirror (string timestamp → cast) ---
    cur.execute("""
        SELECT id, summary, trigger, observer, field_resonance, type, mantra
        FROM collapse_mirror
        WHERE NULLIF(timestamp,'')::timestamptz >= %s
        ORDER BY NULLIF(timestamp,'')::timestamptz DESC
    """, (since_dt,))
    collapse_rows = cur.fetchall()

    # --- Enrichment ---
    cur.execute("""
        SELECT id, collapse_id, enrichment_type, tags, salience, ts
        FROM collapse_enrichment
        WHERE ts::timestamptz >= %s::timestamptz
        ORDER BY ts DESC
    """, (since_dt,))
    enrich_rows = cur.fetchall()

    # --- Chats (random sample within 24h) ---
    cur.execute("""
        SELECT trace_id, prompt, response, created_at
        FROM chat_history_log
        WHERE created_at >= %s
        ORDER BY random()
        LIMIT %s
    """, (since_dt, chat_sample_n))
    chat_rows = cur.fetchall()

    # --- Biometrics summary ---
    bio_summary = _summarize_biometrics(cur, hours=hours, limit_rows=500)

    cur.close()
    conn.close()

    fragments: List[Fragment] = []

    # Collapse
    for r in collapse_rows:
        fragments.append(
            Fragment(
                id=r[0],
                kind="collapse",
                text=(r[1] or r[2] or "").strip(),
                ts=datetime.utcnow().timestamp(),
            )
        )

    # Enrichment
    for r in enrich_rows:
        tags = r[3] or []
        txt = f"{r[2]} {' '.join(tags)}".strip()
        fragments.append(
            Fragment(
                id=r[0],
                kind="enrichment",
                text=txt,
                tags=tags,
                salience=(r[4] or 0.0),
                ts=(r[5].timestamp() if r[5] else datetime.utcnow().timestamp()),
            )
        )

    # Chats (random sample)
    for r in chat_rows:
        prompt = (r[1] or "").strip()
        response = (r[2] or "").strip()
        text = f"User: {prompt}\nOrion: {response}".strip()
        fragments.append(
            Fragment(
                id=r[0],
                kind="chat",
                text=text,
                ts=(r[3].timestamp() if r[3] else datetime.utcnow().timestamp()),
            )
        )

    # Biometrics → single fragment
    if bio_summary:
        bits = []
        if bio_summary["gpu_power_w"] is not None:
            bits.append(f"GPU power={bio_summary['gpu_power_w']}W")
        if bio_summary["gpu_util_pct"] is not None:
            bits.append(f"util={bio_summary['gpu_util_pct']}%")
        if bio_summary["gpu_mem_mb"] is not None:
            bits.append(f"mem={bio_summary['gpu_mem_mb']}MB")
        if bio_summary["cpu_temp_c"] is not None:
            bits.append(f"CPU avg temp={bio_summary['cpu_temp_c']}°C")

        fragments.append(
            Fragment(
                id=f"bio_{uuid.uuid4().hex}",
                kind="biometrics",
                text="System biometrics summary (last 24h): " + (", ".join(bits) if bits else "no data"),
                tags=["gpu", "cpu"],
                ts=datetime.utcnow().timestamp(),
                salience=0.25,
            )
        )

    return fragments
