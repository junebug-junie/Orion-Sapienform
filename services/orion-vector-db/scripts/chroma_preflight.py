#!/usr/bin/env python3
"""
Repair known partial migration state in Chroma 0.4.x SQLite persistence.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


MIGRATION_DIR = "metadb"
MIGRATION_VERSION = 3
MIGRATION_FILE = "00003-full-text-tokenize.sqlite.sql"
FTS_TABLE = "embedding_fulltext_search"


def main(db_path: str) -> int:
    path = Path(db_path)
    if not path.exists():
        print(f"[preflight] no database at {path}, skipping repair")
        return 0

    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (FTS_TABLE,),
        )
        has_fts = cur.fetchone() is not None

        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations' LIMIT 1"
        )
        has_migrations_table = cur.fetchone() is not None

        has_migration = False
        if has_migrations_table:
            cur.execute(
                "SELECT 1 FROM migrations WHERE dir=? AND version=? AND filename=? LIMIT 1",
                (MIGRATION_DIR, MIGRATION_VERSION, MIGRATION_FILE),
            )
            has_migration = cur.fetchone() is not None

        if has_fts and has_migrations_table and not has_migration:
            cur.execute(
                "INSERT INTO migrations(dir, version, filename, sql, hash) VALUES (?, ?, ?, '', '')",
                (MIGRATION_DIR, MIGRATION_VERSION, MIGRATION_FILE),
            )
            conn.commit()
            print("[preflight] repaired missing metadb migration record for existing FTS table")
        else:
            print("[preflight] migration state OK; no repair needed")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/chroma/chroma/chroma.sqlite3"
    raise SystemExit(main(target))
