"""Memory scoreboard: pure parts (issue #3536).

A LongMemEval-style bank of questions about Juniper's real history, scored two ways:

1. **Retrieval** (no model): is a target record id in what the chat prompt would carry? That is
   the DURABLE MEMORY bundle of orion-recall's two chat phases, or the cites of the current core
   page. Bar: 80% of the items whose memory is stored. Recalls that ran degraded (lived memory
   timed out or failed, dense search failed, the recall deadline hit, the RPC failed) are counted
   apart and left out of the score.
2. **Answer** (end to end): the reply to the item, sent as a real Hub chat turn, graded by a local
   model judge into one label. Bars: 70% or more correct, 0 invented, 0 false "no memory" on
   stored items.

Items whose memory is not stored yet (pre-October chat, before the backfill #3537) are reported
apart and never count toward a bar.

Everything here is pure: no DB, bus, Hub or model. ``run_memory_scoreboard.py`` does the I/O. The
bank and every run file are private (Juniper's history) and live host-side only, mode 600. Nothing
in this module prints memory or chat text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA = "memory_scoreboard_bank.v1"
CATEGORIES = ("fact", "event", "time", "update", "abstention")
RETRIEVAL_BAR = 0.80
# A bar is only judged when enough of the bank was measured (degraded / error / unjudged items
# leave the denominator, so a mostly-failed run must not read as a pass).
COVERAGE_FLOOR = 0.80
ANSWER_CORRECT_BAR = 0.70

# The kinds a lived-memory doc id / core-page cite can carry (orion/memory/lived_memory.py KINDS).
TARGET_RE = re.compile(r"^(episode_memory|stance|person_fact):\S+$")
# Any record ref inside a stored core page (orion/memory/pages/juniper.py cites look like
# "[episode_memory:<uuid>]" or "[person_fact:<id>, episode_memory:<uuid>]").
PAGE_REF_RE = re.compile(r"\b((?:episode_memory|person_fact|stance):[A-Za-z0-9_\-]+)")

# Judge labels (what the judge may say) and outcomes (what the scoreboard counts).
JUDGE_LABELS = ("correct", "dont_know", "no_memory", "invented", "unrelated")
# ``no_memory`` is only for backfill items: their memory is not stored yet, so saying so is honest.
OUTCOMES = ("correct", "dont_know", "false_no_memory", "no_memory", "invented", "missed", "unjudged", "error")


# ---------------------------------------------------------------- bank

@dataclass(frozen=True)
class Item:
    qid: str
    category: str
    expected: str
    targets: tuple[str, ...] = ()
    question: Optional[str] = None
    chat_id: Optional[str] = None
    implicit: bool = False
    backfill_pending: bool = False
    superseded: tuple[str, ...] = ()
    wrong_answer: Optional[str] = None
    as_of: Optional[str] = None

    @property
    def stored(self) -> bool:
        """The memory exists in a served store, so retrieval can find it and a denial is false."""
        return bool(self.targets)

    @property
    def kind(self) -> str:
        """``stored`` | ``abstention`` | ``backfill`` -- which bars the item counts toward."""
        if self.category == "abstention":
            return "abstention"
        if self.backfill_pending:
            return "backfill"
        return "stored"


def validate_bank(bank: Mapping[str, Any]) -> List[str]:
    """Every rule the bank must keep. Empty list = valid."""
    errors: List[str] = []
    if bank.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    items = bank.get("items")
    if not isinstance(items, list) or not items:
        return errors + ["items must be a non-empty list"]
    seen: set[str] = set()
    for i, it in enumerate(items):
        qid = str(it.get("qid") or f"#{i}")
        if qid in seen:
            errors.append(f"{qid}: duplicate qid")
        seen.add(qid)
        cat = it.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{qid}: category {cat!r} not in {CATEGORIES}")
        has_q, has_chat = bool(str(it.get("question") or "").strip()), bool(it.get("chat_id"))
        if has_q == has_chat:
            errors.append(f"{qid}: exactly one of question / chat_id")
        if has_chat and not it.get("implicit"):
            errors.append(f"{qid}: a chat_id item is an implicit reference (implicit: true)")
        if not str(it.get("expected") or "").strip():
            errors.append(f"{qid}: expected is empty")
        targets = it.get("targets")
        if not isinstance(targets, list):
            errors.append(f"{qid}: targets must be a list")
            targets = []
        for t in list(targets) + list(it.get("superseded") or []):
            if not TARGET_RE.match(str(t)):
                errors.append(f"{qid}: bad record id {t!r}")
        backfill = bool(it.get("backfill_pending"))
        if cat == "abstention":
            if targets:
                errors.append(f"{qid}: abstention items name no target")
            if backfill:
                errors.append(f"{qid}: abstention items are never backfill_pending")
        elif backfill:
            if targets:
                errors.append(f"{qid}: backfill_pending items have no stored target yet")
            if not it.get("source_chat_ids"):
                errors.append(f"{qid}: backfill_pending items name their source_chat_ids")
        elif not targets:
            errors.append(f"{qid}: a stored item names its target record id(s)")
        if cat == "time" and not it.get("as_of"):
            errors.append(f"{qid}: time items record as_of")
        if cat == "update" and not str(it.get("wrong_answer") or "").strip():
            errors.append(f"{qid}: update items record the wrong (pre-correction) answer")
    return errors


def load_items(bank: Mapping[str, Any]) -> List[Item]:
    errors = validate_bank(bank)
    if errors:
        raise ValueError("invalid memory scoreboard bank: " + "; ".join(errors[:10]))
    out = []
    for it in bank["items"]:
        out.append(Item(
            qid=str(it["qid"]), category=str(it["category"]), expected=str(it["expected"]),
            targets=tuple(it.get("targets") or ()), question=it.get("question") or None,
            chat_id=it.get("chat_id") or None, implicit=bool(it.get("implicit")),
            backfill_pending=bool(it.get("backfill_pending")), superseded=tuple(it.get("superseded") or ()),
            wrong_answer=it.get("wrong_answer") or None, as_of=it.get("as_of") or None,
        ))
    return out


# ---------------------------------------------------------------- retrieval

def page_refs(page_content: Optional[str]) -> set[str]:
    """Record ids a stored core page cites (the page rides in the chat prompt's prefix)."""
    return set(PAGE_REF_RE.findall(str(page_content or "")))


def degraded_reasons(telemetry: Sequence[Mapping[str, Any]], *, lived_timeout_ms: float,
                     rpc_error: Optional[str] = None) -> List[str]:
    """Why one item's recall did not run at full strength (empty = it did).

    ``telemetry`` holds the ``recall_telemetry`` rows the recall service persisted for the item's
    calls (keys ``profile``, ``deadline_hit``, ``backend_counts``, ``timings_ms``). The bus reply
    carries no debug outside diagnostic mode, and diagnostic mode is not part of the
    ``RecallQueryV1`` contract, so the persisted trace is the evidence:

    - any call hit the recall deadline;
    - the continuity call has no ``lived_memory`` backend count (lived memory never started);
    - lived memory ran its exact-word fallback (``backend_counts.lived_memory_degraded``, #3573);
    - lived memory served nothing and waited out its whole budget (it timed out);
    - a call has no trace row at all (cannot be checked, so not trusted).

    Before #3573 a lived-memory error that returned fast read like "nothing fit"; the degraded
    flag now marks it. A turn the skip gate skipped is not degraded: that is the path
    doing what it does.
    """
    reasons: List[str] = []
    if rpc_error:
        reasons.append(f"rpc:{rpc_error}")
    rows = list(telemetry)
    if not rows and not rpc_error:
        reasons.append("trace:missing")
    if any(r.get("deadline_hit") for r in rows):
        reasons.append("recall:deadline_hit")
    for r in rows:
        if r.get("profile") != "chat.continuity.v1":
            continue
        if r.get("lived_status"):
            # #3628: the recall service's own verdict on its lived-memory search. Read it first;
            # the timing heuristics below are only for rows written before that column existed.
            reasons += lived_search_causes(r)
            continue
        counts = _as_dict(r.get("backend_counts"))
        timings = _as_dict(r.get("timings_ms"))
        if "lived_memory" not in counts:
            reasons.append("lived_memory:not_run")
        elif counts.get("lived_memory_degraded"):
            # Step 2 (#3573): the reader failed or timed out and its exact-word fallback served.
            reasons.append("lived_memory:degraded")
        elif not counts.get("lived_memory") and float(timings.get("lived_memory_wait") or 0) >= 0.95 * lived_timeout_ms:
            reasons.append("lived_memory:timeout")
    return reasons


# lived_status values (services/orion-recall/app/lived_memory.py STATUS_*) that mean the lived
# search did not run in full. ok / below_bar are full searches; NULL is a call without lived recall.
SEARCH_NOT_FULL = ("skipped_short", "degraded", "timeout", "error", "off")


def lived_search_causes(row: Mapping[str, Any]) -> List[str]:
    """Why one recall_telemetry row's lived search did not run in full (empty = it did, or the
    row has no lived_status). Causes: ``lived:<status>``, then each ``lived_trace.degraded`` reason
    (``rerank_timeout``, ``embedder_unavailable``, ...) and ``rerank_partial:<scored>/<of>``."""
    status = str(row.get("lived_status") or "")
    trace = _as_dict(row.get("lived_trace"))
    out: List[str] = []
    if status in SEARCH_NOT_FULL:
        out.append(f"lived:{status}")
    part = trace.get("rerank_partial")
    for reason in trace.get("degraded") or []:
        tag = f"lived:{reason}"
        if reason == "rerank_partial" and isinstance(part, Mapping):
            continue   # the dict below says the same thing, with the counts
        if tag not in out:
            out.append(tag)
    if isinstance(part, Mapping):
        # Status ok + this dict: the rerank stopped early and only fit-scored lines served. Not a
        # full search, so it is reported apart, never scored as a clean miss (#3663).
        out.append(f"lived:rerank_partial:{part.get('scored')}/{part.get('of')}")
    return out


_STATUS_TAGS = tuple(f"lived:{st}" for st in SEARCH_NOT_FULL)


def primary_cause(causes: Optional[Sequence[str]]) -> str:
    """The bucket a false denial is reported under: the most specific cause. A lived_trace reason
    (``lived:rerank_timeout``, ``lived:embedder_unavailable``, ``lived:rerank_partial:8/20``) beats
    the bare status (``lived:degraded``), which beats ``recall:*``. ``None`` = the turn's search
    was never checked (``cause:unknown``); an empty list = it ran in full (``search_full``)."""
    if causes is None:
        return "cause:unknown"
    cs = list(causes)
    if not cs:
        return "search_full"
    specific = [c for c in cs if c.startswith("lived:") and c not in _STATUS_TAGS]
    status = [c for c in cs if c in _STATUS_TAGS]
    return (specific or status or cs)[0]


def search_causes(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Why a chat turn's recall did not run in full, from that turn's own recall_telemetry rows
    (#3536 comment). Empty = it ran in full. No rows = ``recall:no_trace`` (the phase-0 skip gate
    fired, or recall never ran). A false "no memory" answer is reported under the first cause, or
    under ``search_full`` when there is none (then the model or the store is at fault, not the search)."""
    if not rows:
        return ["recall:no_trace"]   # the runner says recall:skip_gate when the gate fired
    causes: List[str] = []
    for r in rows:
        if r.get("deadline_hit") and "recall:deadline_hit" not in causes:
            causes.append("recall:deadline_hit")
        for c in lived_search_causes(r):
            if c not in causes:
                causes.append(c)
    return causes


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class RetrievalObservation:
    qid: str
    injected_ids: set[str] = field(default_factory=set)   # DURABLE MEMORY ids, both chat phases
    page_ids: set[str] = field(default_factory=set)       # core page cites
    skipped: bool = False                                 # phase 0 skip gate fired
    degraded: List[str] = field(default_factory=list)


# Search causes that are the chat path working as designed, not a failure: they are scored (a miss
# is a miss) and counted in their own column. lived:skipped_short = a short message with no signal
# for the lived search (lived_memory.py min_query_tokens and the #3613 short-message gate: no known
# entity, and the short rewrite said NONE), which a real turn with that message also does.
SCORED_CAUSES = ("lived:skipped_short",)


def retrieval_verdict(item: Item, obs: RetrievalObservation) -> Dict[str, Any]:
    seen = set(obs.injected_ids) | set(obs.page_ids)
    found = [t for t in item.targets if t in seen]
    degraded = [r for r in obs.degraded if r not in SCORED_CAUSES]
    short = any(r in SCORED_CAUSES for r in obs.degraded)
    return {
        "qid": item.qid, "category": item.category, "kind": item.kind,
        "hit": bool(found), "targets_found": len(found), "targets": len(item.targets),
        "hit_bundle": any(t in obs.injected_ids for t in item.targets),
        "via_page": any(t in obs.page_ids for t in item.targets),
        "superseded_served": [s for s in item.superseded if s in seen],
        "skipped": obs.skipped, "degraded": degraded, "lived_skipped_short": short,
        "injected": len(obs.injected_ids),
    }


def _rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def score_retrieval(items: Sequence[Item], observations: Mapping[str, RetrievalObservation]) -> Dict[str, Any]:
    """Per-category retrieval score. Only stored items score; degraded recalls are counted apart."""
    verdicts = [retrieval_verdict(it, observations[it.qid]) for it in items if it.qid in observations]
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in CATEGORIES:
        vs = [v for v in verdicts if v["category"] == cat]
        stored = [v for v in vs if v["kind"] == "stored"]
        clean = [v for v in stored if not v["degraded"]]
        hits = sum(1 for v in clean if v["hit"])
        bundle_hits = sum(1 for v in clean if v["hit_bundle"])
        by_cat[cat] = {
            "items": len(vs), "stored": len(stored), "scored": len(clean), "hits": hits,
            "hit_rate": _rate(hits, len(clean)), "bundle_hits": bundle_hits,
            "bundle_hit_rate": _rate(bundle_hits, len(clean)),
            "degraded": sum(1 for v in stored if v["degraded"]),
            "lived_skipped_short": sum(1 for v in clean if v["lived_skipped_short"]),
            "lived_skipped_short_hits": sum(1 for v in clean if v["lived_skipped_short"] and v["hit"]),
            "backfill_items": sum(1 for v in vs if v["kind"] == "backfill"),
            "superseded_served": sum(1 for v in vs if v["superseded_served"]),
        }
    # Implicit items (a real past message resent, no direct question) cut across categories:
    # one extra row, not a category, so the per-category rows still add up to the bank.
    imp_q = {it.qid for it in items if it.implicit}
    vs = [v for v in verdicts if v["qid"] in imp_q]
    stored_i = [v for v in vs if v["kind"] == "stored"]
    clean_i = [v for v in stored_i if not v["degraded"]]
    hits_i = sum(1 for v in clean_i if v["hit"])
    bundle_i = sum(1 for v in clean_i if v["hit_bundle"])
    implicit_row = {
        "items": len(vs), "stored": len(stored_i), "scored": len(clean_i), "hits": hits_i,
        "hit_rate": _rate(hits_i, len(clean_i)), "bundle_hits": bundle_i,
        "bundle_hit_rate": _rate(bundle_i, len(clean_i)),
        "degraded": sum(1 for v in stored_i if v["degraded"]),
        "lived_skipped_short": sum(1 for v in clean_i if v["lived_skipped_short"]),
        "lived_skipped_short_hits": sum(1 for v in clean_i if v["lived_skipped_short"] and v["hit"]),
        "backfill_items": sum(1 for v in vs if v["kind"] == "backfill"),
        "superseded_served": sum(1 for v in vs if v["superseded_served"]),
    }
    stored = [v for v in verdicts if v["kind"] == "stored"]
    clean = [v for v in stored if not v["degraded"]]
    hits = sum(1 for v in clean if v["hit"])
    bundle_hits = sum(1 for v in clean if v["hit_bundle"])
    rate = _rate(hits, len(clean))
    return {
        "bar": RETRIEVAL_BAR,
        "stored": len(stored), "scored": len(clean), "hits": hits, "hit_rate": rate,
        # Same, counting only the two recall phases' DURABLE MEMORY (not the core page's cites).
        "bundle_hits": bundle_hits, "bundle_hit_rate": _rate(bundle_hits, len(clean)),
        "coverage": _rate(len(clean), len(stored)),
        "passes": (None if not stored or len(clean) / len(stored) < COVERAGE_FLOOR
                   else rate is not None and rate >= RETRIEVAL_BAR),
        "degraded": [{"qid": v["qid"], "reasons": v["degraded"], "hit": v["hit"]} for v in stored if v["degraded"]],
        "superseded_served": [{"qid": v["qid"], "ids": v["superseded_served"]} for v in verdicts if v["superseded_served"]],
        "backfill": [{"qid": v["qid"], "injected": v["injected"]} for v in verdicts if v["kind"] == "backfill"],
        "skipped_by_gate": [v["qid"] for v in verdicts if v["skipped"]],
        "lived_skipped_short": [{"qid": v["qid"], "hit": v["hit"]} for v in clean if v["lived_skipped_short"]],
        "by_category": by_cat,
        "implicit": implicit_row,
        "items": verdicts,
    }


# ---------------------------------------------------------------- answer (judge)

JUDGE_SYSTEM = (
    "You grade one reply from Orion, an AI, to Juniper, the person who built it. You compare the reply "
    "to what Orion's memory records. You never add facts of your own. You answer with JSON only."
)


def build_judge_prompt(item: Item, message: str, reply: str, record_texts: Sequence[str] = ()) -> str:
    """The judge's user message. Lives only in memory and goes only to a local lane."""
    if item.kind == "abstention":
        truth = ("NOTHING is recorded about this. It never happened, or Orion was never told. "
                 "A good reply says it does not know, has no memory of it, or that it did not happen.")
    else:
        truth = item.expected
        if record_texts:
            truth += "\nRecord lines:\n" + "\n".join(f"- {t}" for t in record_texts[:6])
    extra = ""
    if item.wrong_answer and item.kind != "abstention":
        extra = (f"\nThis was corrected. The OLD, wrong answer was: {item.wrong_answer}. A reply that gives "
                 "the old answer as true is 'invented'.")
    if item.as_of:
        extra += (f"\nThis question was written on {item.as_of}. Relative dates in it refer to that day; "
                  "a reply that gives the right dated event counts as correct.")
    if item.implicit:
        extra += ("\nJuniper's message is not a direct question. It leans on shared past. 'correct' means the "
                  "reply shows Orion knows the recorded truth, or uses it. A friendly reply that neither "
                  "shows nor contradicts it is 'unrelated'.")
    return (
        f"Juniper's message:\n{message}\n\n"
        f"What Orion's memory records (the truth):\n{truth}{extra}\n\n"
        f"Orion's reply:\n{reply}\n\n"
        "Pick exactly one label:\n"
        "- correct: the reply gives the recorded truth (for a never-happened item: says it does not know, "
        "has no record, or that it did not happen). Minor wording differences are fine, and so is a date one "
        "day off (records are stored in UTC; Juniper lives in US Mountain time).\n"
        "- dont_know: the reply says it is not sure or asks to be reminded, but does not say it never happened.\n"
        "- no_memory: the reply says it has no memory or record of this, or that it did not happen.\n"
        "- invented: the reply CONTRADICTS the recorded truth, gets the asked-for answer wrong, describes a "
        "never-happened event as real, or gives a corrected old answer as true.\n"
        "- unrelated: the reply does not address it.\n"
        "The record lines above are only the few records this question targets, not all of Orion's memory. "
        "Extra details the record lines do not mention are NOT invented; Orion may know them from other "
        "memories. Only a detail that contradicts the recorded truth, or a wrong answer to what was asked, "
        "makes a reply invented. A detail contradicts only when the truth says something DIFFERENT about the "
        "same thing (another date for the same event, another answer to the same question). Leaving parts of "
        "the truth out is not invented: if what the reply says agrees with the truth and gets its main point, "
        "it is correct.\n"
        'Answer with JSON only: {"label": "<one label>", "why": "<one short sentence>"}'
    )


_JSON_OBJ = re.compile(r"\{.*?\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_judge_label(raw: str) -> Optional[str]:
    """The judge's label, or None when it gave none we can read."""
    text = _THINK.sub("", str(raw or "")).strip()
    for m in _JSON_OBJ.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        label = str(obj.get("label") or "").strip().lower().replace(" ", "_").replace("'", "")
        if label in JUDGE_LABELS:
            return label
    m = re.search(r"\blabel\s*[:=]\s*\"?([a-z_' ]+?)\"?(?:[\s,.}]|$)", text.lower())
    if m:
        label = m.group(1).strip().replace(" ", "_").replace("'", "")
        if label in JUDGE_LABELS:
            return label
    return None


def outcome_for(item: Item, label: Optional[str]) -> str:
    """Map the judge's label to what the scoreboard counts, by item kind."""
    if label is None:
        return "unjudged"
    if label == "unrelated":
        return "missed"
    if item.kind == "abstention":
        return "invented" if label == "invented" else "correct"
    if label == "no_memory":
        return "no_memory" if item.kind == "backfill" else "false_no_memory"
    return label  # correct | dont_know | invented


def _answer_row(items: Iterable[Item], outcomes: Mapping[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {o: 0 for o in OUTCOMES}
    row["items"] = 0
    row["backfill"] = {o: 0 for o in OUTCOMES}
    for it in items:
        if it.qid not in outcomes:
            continue
        o = outcomes[it.qid]
        if it.kind == "backfill":
            row["backfill"][o] += 1
            continue
        row["items"] += 1
        row[o] += 1
    measured = row["items"] - row["error"] - row["unjudged"]
    row["correct_rate"] = _rate(row["correct"], measured)
    return row


def score_answers(items: Sequence[Item], outcomes: Mapping[str, str],
                  causes: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Per-category answer score. Scored = stored + abstention items; backfill items apart.
    Hub failures (``error``) and unreadable judge output (``unjudged``) are counted apart too.

    ``causes`` maps a qid to its turn's :func:`search_causes`. A false "no memory" on a stored item
    is always a false denial; it is also reported under the first cause (``search_full`` when the
    search ran in full), so a denial after a skipped or degraded search is visible as such."""
    by_cat = {cat: _answer_row([it for it in items if it.category == cat], outcomes) for cat in CATEGORIES}
    implicit_row = _answer_row([it for it in items if it.implicit], outcomes)
    scored = [it for it in items if it.qid in outcomes and it.kind != "backfill"]
    measured = [it for it in scored if outcomes[it.qid] not in ("error", "unjudged")]
    correct = sum(1 for it in measured if outcomes[it.qid] == "correct")
    invented = [it.qid for it in measured if outcomes[it.qid] == "invented"]
    false_no = [it.qid for it in measured if it.kind == "stored" and outcomes[it.qid] == "false_no_memory"]
    rate = _rate(correct, len(measured))
    by_cause: Dict[str, List[str]] = {}
    for qid in false_no:
        by_cause.setdefault(primary_cause((causes or {}).get(qid)), []).append(qid)
    return {
        "false_no_memory_by_cause": by_cause,
        "false_no_memory_causes": {q: (list((causes or {}).get(q)) if (causes or {}).get(q) is not None else None)
                                   for q in false_no},
        "bars": {"correct_rate": ANSWER_CORRECT_BAR, "invented": 0, "false_no_memory": 0},
        "scored": len(scored), "measured": len(measured), "correct": correct, "correct_rate": rate,
        "invented": invented, "false_no_memory": false_no,
        "coverage": _rate(len(measured), len(scored)),
        "passes": ({k: None for k in ("correct_rate", "invented", "false_no_memory")}
                   if not scored or len(measured) / len(scored) < COVERAGE_FLOOR else {
            "correct_rate": rate is not None and rate >= ANSWER_CORRECT_BAR,
            "invented": not invented,
            "false_no_memory": not false_no,
        }),
        "not_measured": [{"qid": it.qid, "outcome": outcomes[it.qid]} for it in scored
                         if outcomes[it.qid] in ("error", "unjudged")],
        "backfill": [{"qid": it.qid, "outcome": outcomes[it.qid]} for it in items
                     if it.qid in outcomes and it.kind == "backfill"],
        "by_category": by_cat,
        "implicit": implicit_row,
    }


# ---------------------------------------------------------------- report

def _verdict(passed: Optional[bool]) -> str:
    return "NOT MEASURED (coverage below floor)" if passed is None else ("PASS" if passed else "FAIL")


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{round(100 * x)}%"


def render_table(retrieval: Optional[Mapping[str, Any]], answers: Optional[Mapping[str, Any]],
                 *, run_date: str, bank_items: int) -> str:
    """Markdown per-category table. Counts and qids only, never memory or chat text."""
    lines = [f"# Memory scoreboard {run_date}", "", f"Bank: {bank_items} items.", ""]
    if retrieval is not None:
        lines += [
            f"## Retrieval (bar {_pct(RETRIEVAL_BAR)} of stored items, degraded recalls apart)", "",
            "| category | items | stored | scored | hits | hit rate | recall-only hit rate | too short for lived search (hits) | degraded | backfill items | superseded served |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        def _short(r: Mapping[str, Any]) -> str:
            return f"{r.get('lived_skipped_short', 0)} ({r.get('lived_skipped_short_hits', 0)})"

        rows = list(retrieval["by_category"].items())
        if retrieval.get("implicit"):
            rows.append(("implicit (subset of the above)", retrieval["implicit"]))
        for cat, r in rows:
            lines.append(f"| {cat} | {r['items']} | {r['stored']} | {r['scored']} | {r['hits']} | "
                         f"{_pct(r['hit_rate'])} | {_pct(r['bundle_hit_rate'])} | {_short(r)} | {r['degraded']} | "
                         f"{r['backfill_items']} | {r['superseded_served']} |")
        short = retrieval.get("lived_skipped_short") or []
        lines.append(f"| **all** | | {retrieval['stored']} | {retrieval['scored']} | {retrieval['hits']} | "
                     f"**{_pct(retrieval['hit_rate'])}** | {_pct(retrieval['bundle_hit_rate'])} | "
                     f"{len(short)} ({sum(1 for x in short if x['hit'])}) | "
                     f"{len(retrieval['degraded'])} | {len(retrieval['backfill'])} | {len(retrieval['superseded_served'])} |")
        lines += ["", "Hit = a target id is in DURABLE MEMORY (both chat recall phases) or cited by the core page. "
                  "Recall-only = DURABLE MEMORY alone. "
                  "Too short = the lived search skipped the message for too few content words (the real chat "
                  "path does the same, so these are scored).", "",
                  f"Retrieval bar: {_verdict(retrieval['passes'])} (coverage {_pct(retrieval['coverage'])}).", ""]
        if retrieval["degraded"]:
            lines.append("Degraded recalls (not scored): " + ", ".join(
                f"{d['qid']} ({'/'.join(d['reasons'])})" for d in retrieval["degraded"]))
            lines.append("")
        if retrieval["skipped_by_gate"]:
            lines += ["Skipped by the phase-0 gate (no recall ran; only core-page cites can hit): " + ", ".join(retrieval["skipped_by_gate"]), ""]
    if answers is not None:
        lines += [
            f"## Answer (bars: correct >= {_pct(ANSWER_CORRECT_BAR)}, 0 invented, 0 false 'no memory' on stored)", "",
            "| category | items | correct | don't know | invented | false no-memory | missed | not measured | correct rate |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for cat, r in answers["by_category"].items():
            lines.append(f"| {cat} | {r['items']} | {r['correct']} | {r['dont_know']} | {r['invented']} | "
                         f"{r['false_no_memory']} | {r['missed']} | {r['error'] + r['unjudged']} | {_pct(r['correct_rate'])} |")
        r = answers.get("implicit")
        if r:
            lines.append(f"| implicit (subset of the above) | {r['items']} | {r['correct']} | {r['dont_know']} | {r['invented']} | "
                         f"{r['false_no_memory']} | {r['missed']} | {r['error'] + r['unjudged']} | {_pct(r['correct_rate'])} |")
        lines.append(f"| **all** | {answers['scored']} | {answers['correct']} | | {len(answers['invented'])} | "
                     f"{len(answers['false_no_memory'])} | | {len(answers['not_measured'])} | **{_pct(answers['correct_rate'])}** |")
        p = answers["passes"]
        lines += ["", "Answer bars: correct " + _verdict(p["correct_rate"]) + ", invented " + _verdict(p["invented"])
                  + ", false no-memory " + _verdict(p["false_no_memory"])
                  + f" (coverage {_pct(answers['coverage'])}).", ""]
        if answers["invented"]:
            lines += ["Invented: " + ", ".join(answers["invented"]), ""]
        if answers["false_no_memory"]:
            lines += ["False no-memory: " + ", ".join(answers["false_no_memory"]), ""]
            by_cause = answers.get("false_no_memory_by_cause") or {}
            lines += ["False no-memory by cause (search_full = the search ran in full): " + "; ".join(
                f"{c} {len(q)} ({', '.join(q)})" for c, q in sorted(by_cause.items())), ""]
    backfill_r = (retrieval or {}).get("backfill") or []
    backfill_a = (answers or {}).get("backfill") or []
    if backfill_r or backfill_a:
        lines += ["## Not stored yet (pre-October, waits on backfill #3537; no bar)", ""]
        outs = {b["qid"]: b["outcome"] for b in backfill_a}
        for qid in sorted({b["qid"] for b in backfill_r} | set(outs)):
            lines.append(f"- {qid}: answer {outs.get(qid, '-')}")
        lines.append("")
    return "\n".join(lines)


def summary_line(run_date: str, retrieval: Optional[Mapping[str, Any]], answers: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """One line for the weekly history file (counts only)."""
    out: Dict[str, Any] = {"date": run_date}
    if retrieval is not None:
        out["retrieval"] = {"hit_rate": retrieval["hit_rate"], "bundle_hit_rate": retrieval["bundle_hit_rate"],
                            "scored": retrieval["scored"],
                            "degraded": len(retrieval["degraded"]),
                            "by_category": {c: r["hit_rate"] for c, r in retrieval["by_category"].items()},
                            "implicit": (retrieval.get("implicit") or {}).get("hit_rate")}
    if answers is not None:
        out["answer"] = {"correct_rate": answers["correct_rate"], "measured": answers["measured"],
                         "invented": len(answers["invented"]), "false_no_memory": len(answers["false_no_memory"]),
                         "false_no_memory_by_cause": {c: len(q) for c, q in (answers.get("false_no_memory_by_cause") or {}).items()},
                         "by_category": {c: r["correct_rate"] for c, r in answers["by_category"].items()}}
    return out


def iter_record_ids(items: Iterable[Item]) -> List[str]:
    """Every target and superseded id the bank names (the runner reads their text for the judge)."""
    ids: List[str] = []
    for it in items:
        for t in (*it.targets, *it.superseded):
            if t not in ids:
                ids.append(t)
    return ids
