#!/usr/bin/env python3
"""Operator distiller for memory cards (Phase 5 — scaffold)."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Distill memory cards from transcripts (Phase 5 scaffold).")
    p.add_argument("--transcript", type=str, required=True)
    p.add_argument("--project", type=str, default="")
    p.add_argument("--since", type=str, default="")
    p.add_argument("--today", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("use only one of --apply or --dry-run")
    mode = "apply" if args.apply else "dry-run"
    print(f"distill_memory_cards: {mode} transcript={args.transcript!r} (scaffold; wire DAL in Phase 5)")


if __name__ == "__main__":
    main()
