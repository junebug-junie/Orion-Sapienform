"""Episode distiller plumbing shared by the durable graph and the offline eval.

Load an episode's turns (full text), render ``memory_episode_distill.j2``, parse the model's JSON.
No judgment about what to remember lives here; that is the distiller's job, and
``orion.memory.episode.validate`` checks what it returns.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from orion.memory.episode.validate import DEFAULT_TZ, EpisodeTurn
from orion.schemas.memory_episode import EpisodeDistillationV1

PROMPT_PATH = Path(__file__).resolve().parents[2] / "cognition" / "prompts" / "memory_episode_distill.j2"
# The template states its own version in its first line. Templates from before the marker existed
# (v1/v2) have none; v2 is the last of those.
_VERSION_MARKER = re.compile(r"\{#-?\s*prompt_version:\s*(\S+?)\s*-?#\}")
UNMARKED_TEMPLATE_VERSION = "memory_episode_distill.v2"


def template_prompt_version(path: Path | None = None) -> str:
    """The version of the template that ``render_prompt`` renders (default: the current PROMPT_PATH).

    The durable graph stamps THIS on the run, not the brief's ``prompt_version``: the brief is built
    by memory-consolidation and can come from a different image than the one rendering the prompt.
    """
    head = (path or PROMPT_PATH).read_text(encoding="utf-8")[:300]
    m = _VERSION_MARKER.search(head)
    return m.group(1) if m else UNMARKED_TEMPLATE_VERSION



# A workflow-command turn is identified by the Hub workflow runtime's own reply header
# ("Workflow: Journal Pass", "Workflow 'github_compactor_pass' ..."): a structural marker the
# runtime writes, per the spec's skip rule -- not a judgment of what is memorable.
_WORKFLOW_REPLY = re.compile(r"^\s*Workflow\b")

LOAD_TURNS_SQL = """
SELECT correlation_id, prompt, response, created_at
FROM chat_history_log
WHERE correlation_id = ANY(%s)
ORDER BY created_at ASC
"""


def is_workflow_command_reply(response: str | None) -> bool:
    return bool(_WORKFLOW_REPLY.match(str(response or "")))


def _utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def turns_from_rows(rows: Iterable[dict[str, Any]]) -> list[EpisodeTurn]:
    """Rows (correlation_id, prompt, response, created_at) -> labelled turns t1..tN in time order."""
    ordered = sorted(rows, key=lambda r: (_utc(r.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)))
    return [
        EpisodeTurn(
            label=f"t{i}",
            correlation_id=str(r["correlation_id"]),
            prompt=str(r.get("prompt") or ""),
            response=str(r.get("response") or ""),
            created_at=_utc(r.get("created_at")),
            is_command=is_workflow_command_reply(r.get("response")),
        )
        for i, r in enumerate(ordered, start=1)
    ]


def turns_to_state(turns: list[EpisodeTurn]) -> list[dict[str, Any]]:
    return [
        {
            "label": t.label,
            "correlation_id": t.correlation_id,
            "prompt": t.prompt,
            "response": t.response,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "is_command": t.is_command,
        }
        for t in turns
    ]


def turns_from_state(items: list[dict[str, Any]]) -> list[EpisodeTurn]:
    return [
        EpisodeTurn(
            label=str(i["label"]),
            correlation_id=str(i["correlation_id"]),
            prompt=str(i.get("prompt") or ""),
            response=str(i.get("response") or ""),
            created_at=_utc(i.get("created_at")),
            is_command=bool(i.get("is_command")),
        )
        for i in items
    ]


def render_prompt(
    *,
    episode_id: str,
    turns: list[EpisodeTurn],
    candidate_referents: Iterable[str] = (),
    tz_name: str = DEFAULT_TZ,
) -> str:
    from jinja2 import Environment, StrictUndefined

    tz = ZoneInfo(tz_name)

    def local(dt: Optional[datetime]) -> str:
        return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M") if dt else "unknown time"

    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
    template = env.from_string(PROMPT_PATH.read_text(encoding="utf-8"))
    times = [t.created_at for t in turns if t.created_at]
    return template.render(
        episode_id=episode_id,
        timezone=tz_name,
        started_local=local(min(times) if times else None),
        ended_local=local(max(times) if times else None),
        candidate_referents=sorted(set(candidate_referents)),
        turns=[
            {
                "label": t.label,
                "local_time": local(t.created_at),
                "is_command": t.is_command,
                "prompt": t.prompt,
                "response": t.response,
            }
            for t in turns
        ],
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_distillation(text: str) -> EpisodeDistillationV1:
    """The model's answer -> EpisodeDistillationV1. Raises ValueError when no JSON object parses
    (an attempt failure for the durable graph, never an empty success)."""
    raw = str(text or "").strip()
    raw = _FENCE.sub("", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("distill_no_json_object")
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"distill_json_invalid:{exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("distill_json_not_object")
    # Drop individual malformed items rather than the whole answer; the validator logs what is kept.
    memories, questions = [], []
    from orion.schemas.memory_episode import DistilledMemoryV1, DistilledQuestionV1

    for item in data.get("memories") or []:
        try:
            memories.append(DistilledMemoryV1.model_validate(item))
        except Exception:  # noqa: BLE001 -- counted by malformed_item_count
            continue
    for item in data.get("questions") or []:
        try:
            questions.append(DistilledQuestionV1.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    return EpisodeDistillationV1(memories=memories, questions=questions)


def malformed_item_count(text: str) -> int:
    """How many memory/question items in the raw answer failed the schema (reported, not stored)."""
    try:
        raw = _FENCE.sub("", str(text or "").strip()).strip()
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:  # noqa: BLE001
        return 0
    from orion.schemas.memory_episode import DistilledMemoryV1, DistilledQuestionV1

    bad = 0
    for model, key in ((DistilledMemoryV1, "memories"), (DistilledQuestionV1, "questions")):
        for item in data.get(key) or []:
            try:
                model.model_validate(item)
            except Exception:  # noqa: BLE001
                bad += 1
    return bad
