"""The real memory store behind the compaction applier: reverie-thought salience, downscale only.

What it changes: ``substrate_reverie_thought.salience`` of thoughts named by a delta's downscale entries
(``target_id = "substrate_reverie_thought:<thought_id>"``), and nothing else. The REM pass proposes those
entries for the thoughts of reverie chains that settled and were folded into a consolidate entry
(``app.rem_compaction.downscale_entries``). Salience is what Orion's day reading ranks reverie thoughts by
(``orion/orion_day/budget.py``), so a downscaled settled thought gives way to an unsettled one.

The original value is never lost: the reverie writer stores it twice, in the ``salience`` column and in
``thought_json.salience``. This store writes only the column. The snapshot file holds the column's
before-state too, and ``restore`` (or ``python -m app.compaction_store restore <snapshot_before.json>``)
puts it back.

Guards, each a hard refusal (``StoreRefused``), so the applier rolls back:
  * a target that is not a reverie thought;
  * a ``new_w`` above ``old_w`` (downscale never raises a value) or below 0;
  * a row whose salience is no longer ``old_w`` (compare-and-set: the delta is stale);
  * ``prune``: irreversible, always refused here, whatever consent says;
  * ``write_gist``: this store has no gist-card table (``supports_gists = False``); the applier skips
    consolidate entries and counts them in the receipt instead of calling this.
"""
from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from typing import Any, Optional

from orion.schemas.compaction import MemoryCompactionDeltaV1

logger = logging.getLogger("orion-dream.compaction_store")

TABLE = "substrate_reverie_thought"
COLUMN = "salience"
TARGET_PREFIX = f"{TABLE}:"
SNAPSHOT_SCHEMA = "dream.compaction.snapshot.v1"
# Salience is double precision; the delta carries the exact value read, so equality is exact in practice.
EPS = 1e-9


class StoreRefused(RuntimeError):
    """The store will not do this. Raised inside an apply, so the applier restores the snapshot."""


def thought_id_of(target_id: str) -> str:
    """The reverie thought id a downscale target names. Anything else is refused."""
    if not isinstance(target_id, str) or not target_id.startswith(TARGET_PREFIX):
        raise StoreRefused(f"not a reverie-thought target: {target_id!r}")
    tid = target_id[len(TARGET_PREFIX):]
    if not tid:
        raise StoreRefused("empty reverie-thought id")
    return tid


class ReverieSalienceStore:
    """``app.compaction_applier.CompactionMemoryStore`` over ``substrate_reverie_thought.salience``."""

    supports_gists = False

    def __init__(self, engine: Any = None, *, uri: Optional[str] = None):
        self._engine = engine
        self._uri = uri

    def engine(self):
        if self._engine is None:
            from sqlalchemy import create_engine

            if not self._uri:
                from app.settings import settings

                self._uri = settings.POSTGRES_URI
            self._engine = create_engine(self._uri, pool_pre_ping=True)
        return self._engine

    def snapshot(self, delta: MemoryCompactionDeltaV1) -> dict[str, Any]:
        """The current salience of every downscale target. A missing row fails the snapshot (fail closed:
        the applier does not apply what it cannot put back)."""
        from sqlalchemy import bindparam, text

        ids = list(dict.fromkeys(thought_id_of(e.target_id) for e in delta.downscale))
        rows: dict[str, float] = {}
        if ids:
            stmt = text(f"SELECT thought_id, {COLUMN} FROM {TABLE} WHERE thought_id IN :ids").bindparams(
                bindparam("ids", expanding=True))
            with self.engine().connect() as conn:
                for tid, val in conn.execute(stmt, {"ids": ids}).all():
                    rows[str(tid)] = None if val is None else float(val)
        missing = [t for t in ids if t not in rows]
        if missing:
            raise StoreRefused(f"snapshot: {len(missing)} target(s) not found, e.g. {missing[0]}")
        return {"schema": SNAPSHOT_SCHEMA, "table": TABLE, "column": COLUMN, "delta_id": delta.delta_id,
                "rows": rows}

    def restore(self, snapshot: dict[str, Any]) -> int:
        """Put every snapshotted salience back, in one transaction. Returns the rows restored."""
        from sqlalchemy import text

        if (snapshot.get("schema"), snapshot.get("table"), snapshot.get("column")) != (SNAPSHOT_SCHEMA, TABLE, COLUMN):
            raise StoreRefused("restore: not a reverie-salience snapshot")
        rows = snapshot.get("rows") or {}
        n = 0
        with self.engine().begin() as conn:
            for tid, val in rows.items():
                n += conn.execute(text(f"UPDATE {TABLE} SET {COLUMN} = :v WHERE thought_id = :id"),
                                  {"v": val, "id": str(tid)}).rowcount
        logger.info("compaction_store restored delta=%s rows=%d", snapshot.get("delta_id"), n)
        return n

    def downscale(self, target_id: str, old_w: float, new_w: float) -> None:
        from sqlalchemy import text

        tid = thought_id_of(target_id)
        old_w, new_w = float(old_w), float(new_w)
        if not (0.0 <= new_w <= old_w):
            raise StoreRefused(f"downscale must lower the value: {old_w} -> {new_w}")
        with self.engine().begin() as conn:
            n = conn.execute(text(f"UPDATE {TABLE} SET {COLUMN} = :new WHERE thought_id = :id "
                                  f"AND abs({COLUMN} - :old) < :eps"),
                             {"new": new_w, "old": old_w, "id": tid, "eps": EPS}).rowcount
        if n != 1:
            raise StoreRefused(f"stale target {tid}: salience is no longer {old_w} (updated {n} rows)")

    def prune(self, episodic_id: str) -> None:
        raise StoreRefused("prune is irreversible and refused by this store")

    def write_gist(self, gist_card: str, evidence_refs: list[str], supersedes: list[str]) -> None:
        raise StoreRefused("this store has no gist-card table")


@lru_cache(maxsize=1)
def default_store() -> ReverieSalienceStore:
    """The store the REM pass hands the applier (engine created on first use, not at import)."""
    return ReverieSalienceStore()


def restore_snapshot_file(path: str, store: Optional[ReverieSalienceStore] = None) -> int:
    """Operator rollback of one applied delta: restore from its ``snapshot_before.json``."""
    with open(path, encoding="utf-8") as fh:
        snap = json.load(fh)
    return (store or default_store()).restore(snap)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Restore reverie salience from a compaction snapshot.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("restore")
    r.add_argument("snapshot_path")
    a = ap.parse_args(argv)
    n = restore_snapshot_file(a.snapshot_path)
    print(f"restored {n} row(s) from {a.snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
