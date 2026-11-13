#!/usr/bin/env python3
"""Recheck a saved replay without services or model calls; output may contain private evidence.

Usage: PYTHONPATH=. python services/orion-hub/evals/recheck_reading_resonance_archive.py /tmp/replay
The archive must contain packets.jsonl, prompts/<packet_id>.txt, and judgments/*.json.
This checks structure and citations, not whether a connection is worth sharing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from orion.schemas.reading_resonance import ResonancePacketV1
from orion.world_pulse_read.resonance import check_judgment


def recheck(root: Path) -> dict:
    judgments = {}
    for path in sorted((root / "judgments").glob("*.json")):
        item = json.loads(path.read_text())
        if item["packet_id"] in judgments:
            raise ValueError("Multiple judgments for one packet; select one frozen run first")
        judgments[item["packet_id"]] = item
    results = []
    for line in (root / "packets.jsonl").read_text().splitlines():
        packet = ResonancePacketV1.model_validate_json(line)
        prompt = (root / "prompts" / f"{packet.packet_id}.txt").read_text()
        retained = [tuple(line.removeprefix("retained text ").split(": ", 1))
                    for line in prompt.splitlines() if line.startswith("retained text ")]
        cached = judgments[packet.packet_id]
        judgment, check = check_judgment(packet, cached["raw"], retained_texts=retained, prompt=prompt)
        results.append({"item": len(results), "packet_id": packet.packet_id,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "raw_sha256": hashlib.sha256(cached["raw"].encode()).hexdigest(),
                        "check": check.model_dump(mode="json"),
                        "judgment": judgment.model_dump(mode="json") if judgment else None})
    return {"count": len(results), "accepted": sum(r["check"]["accepted"] for r in results),
            "decisions": dict(Counter(r["check"]["decision"] for r in results)), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(recheck(args.archive), indent=2))
