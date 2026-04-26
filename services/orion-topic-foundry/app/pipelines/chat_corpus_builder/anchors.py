from __future__ import annotations

import re
from collections import defaultdict

from .types import AnchorRecord, ChatTurnRecord

SERVICE_RE = re.compile(r"\borion-[a-z0-9-]+\b", re.IGNORECASE)
TABLE_RE = re.compile(r"\b[a-z][a-z0-9_]+_(index|log|entry|history)\b")
PATH_RE = re.compile(r"\b(?:services|orion)/[A-Za-z0-9_./-]+\b")
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
COMMAND_RE = re.compile(r"(^|\n)\s*(docker compose[^\n]*|pytest[^\n]*|python[^\n]*|psql[^\n]*)", re.IGNORECASE)
ERROR_RE = re.compile(r"(input device is not a tty|candidate_count=0|traceback[^\n]*)", re.IGNORECASE)
URL_RE = re.compile(r"(/healthz|/corpora/[a-z_/]+)")
ARTIFACT_RE = re.compile(r"\b(PageIndex|GraphDB|journaler|Topic Foundry)\b", re.IGNORECASE)


def extract_anchors(turns: list[ChatTurnRecord]) -> list[AnchorRecord]:
    out: list[AnchorRecord] = []
    for turn in turns:
        text = f"{turn.prompt}\n{turn.response}\n{turn.thought_process}"
        by_type: dict[str, set[str]] = defaultdict(set)
        for regex, key in (
            (SERVICE_RE, "service"),
            (TABLE_RE, "table"),
            (PATH_RE, "path"),
            (ENV_RE, "env"),
            (COMMAND_RE, "command"),
            (ERROR_RE, "error"),
            (URL_RE, "url"),
            (ARTIFACT_RE, "artifact"),
        ):
            for m in regex.finditer(text):
                value = _clean(m.group(0))
                if value:
                    by_type[key].add(value)
        anchors: list[str] = []
        types: list[str] = []
        surfaces: list[str] = []
        for anchor_type in sorted(by_type):
            for anchor in sorted(by_type[anchor_type]):
                anchors.append(anchor.lower())
                types.append(anchor_type)
                surfaces.append(anchor)
        out.append(AnchorRecord(turn_id=turn.turn_id, anchors=anchors, anchor_types=types, surface_forms=surfaces))
    return out


def _clean(value: str) -> str:
    return " ".join(value.strip().split())
