import psycopg2
from datetime import datetime, timedelta
from app.utils import Fragment
from app.settings import settings

def fetch_recent_sql_fragments(hours: int = 24):
    conn = psycopg2.connect(settings.POSTGRES_URI)
    cur = conn.cursor()
    since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

    cur.execute("""
        SELECT id, summary, trigger, observer, field_resonance, type, mantra
        FROM collapse_mirror WHERE timestamp >= %s
    """, (since,))
    collapse_rows = cur.fetchall()

    cur.execute("""
        SELECT id, collapse_id, enrichment_type, tags, salience
        FROM collapse_enrichment WHERE ts >= %s
    """, (since,))
    enrich_rows = cur.fetchall()

    cur.execute("""
        SELECT trace_id, prompt, response, created_at
        FROM chat_history_log WHERE created_at >= %s
    """, (since,))
    chat_rows = cur.fetchall()

    cur.close(); conn.close()

    fragments = []
    for r in collapse_rows:
        fragments.append(Fragment(id=r[0], kind="collapse", text=r[1] or r[2] or "", ts=datetime.utcnow().timestamp()))
    for r in enrich_rows:
        txt = f"{r[2]} {' '.join(r[3] or [])}"
        fragments.append(Fragment(id=r[0], kind="enrichment", text=txt, tags=r[3] or []))
    for r in chat_rows:
        text = f"User: {r[1]}\nOrion: {r[2]}"
        fragments.append(Fragment(id=r[0], kind="chat", text=text))
    return fragments
