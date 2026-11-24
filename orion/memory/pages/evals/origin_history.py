"""page:origin history eval (issue #3733): is "how I came to be" true, cited honestly, and steady?

Read-only. Scores every stored ``page:origin`` version in ``self_concept_history`` (and, optionally, a
dry-run page from ``scripts/core_page_dry_run.py --page origin --out <json>``):

1. **Loss sentence** (Decision 3 of the core-pages spec): the live check ``check_origin_page`` is
   re-run on the stored page, with the cited records as its sources. ``loss_stated`` is True when
   some sentence states the memory loss and cites a record of it. ``check_ok`` is the live check
   against the CITED records only; ``loss_dropped`` is True when today's pool holds a loss record and
   the version states no loss (a version that drops the loss and its cite would otherwise pass).
2. **Cite mix**: the version's ``evidence_refs`` counted by kind (stance / self-definition /
   episode_memory), next to the pool the fold is offered today (``sources_from_rows`` on the live
   source SQL). The pool at each night's fold is not stored, so the pool is today's, not then's.
3. **Stability**: word-level similarity between consecutive versions (``difflib``), and which
   sentences (claims) appear or disappear. A sentence is "kept" when the next version holds a
   sentence at least ``SAME_CLAIM`` similar to it.
4. **Cite faithfulness**, deterministic first pass: content-word overlap between each sentence and
   the records it cites. A sentence is ``unsupported`` when it shares fewer than ``MIN_SHARED``
   content words with all its cited records together; ``weak`` when it shares under ``WEAK_RATIO`` of
   its own content words; ``no_record`` when it cites only records that no longer exist; ``uncited``
   when it cites nothing. Dates ("August 13", "August 13th") must match a cited record's UTC or
   Denver day or appear in its text (month and day only, the year is not checked); numbers must
   appear in a cited record's text. Stemming is crude (died/dies, lost/lose stay apart). A single cite is ``unrelated`` when it shares fewer than ``MIN_SHARED`` content
   words with the sentence (run 1 cited an unrelated AC record next to a real one). A floor, not a
   judge: a paraphrase with no shared words reads as unsupported, and a sentence that reuses a
   record's words in a wrong claim reads as supported.
5. **Self-sense** (optional): ``self_sense_eval_log`` ``what_are_you`` answers before and after the
   first page:origin version, in equal windows of ``--self-sense-days`` days each side: mean scores,
   and how often an answer states a memory loss. Other changes in the same window (page:juniper and
   page:lately went live the same week) are not separated out.

``score_*`` / ``summarize_*`` functions are pure and unit-tested. ``main`` does the read-only DB pass.

Privacy: the page quotes Juniper's private messages and this repo is public. stdout carries COUNTS
ONLY. Sentences, record text and ids go only to ``--out`` (default under ``~/.orion/evals/``, never
in the repo).

    .venv/bin/python -m orion.memory.pages.evals.origin_history [--page-json <dry_run.json>] [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from orion.memory.pages.fold import (
    EPISODE_MEMORY_KIND, REF_KINDS, SELF_CONCEPT_KIND, STANCE_KIND, PageSource, _norm, render_text,
)
from orion.memory.pages.origin import _LOSS_STATED, PAGE_ORIGIN, check_origin_page, memory_loss_refs

REPO = Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = Path.home() / ".orion" / "evals"
ZONE = ZoneInfo("America/Denver")

# Two sentences say the same claim at this word-level similarity or above.
SAME_CLAIM = 0.6
# A sentence needs this many content words in common with its cited records to count as supported.
MIN_SHARED = 2
# ... and at least this share of its own content words (below it: weak, still counted as supported).
WEAK_RATIO = 0.25

# A stored cite group: "[stance:abc, self_concept_history:xyz]".
_STORED_GROUP = re.compile(
    rf"\[\s*((?:{'|'.join(REF_KINDS)}):[^\]\s,;]+(?:\s*[,;]\s*(?:{'|'.join(REF_KINDS)}):[^\]\s,;]+)*)\s*\]"
)
_REF = re.compile(rf"(?:{'|'.join(REF_KINDS)}):[^\]\s,;]+")

# Words that carry no claim on this page: function words, the two names (every sentence is about
# Orion and Juniper), and page boilerplate. Kept small and plain on purpose.
_STOP = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each even ever every few for from further
had has have having her here hers herself him his how i if in into is it its itself just like me more
most much my myself no nor not now of off on once one only or other our ours out over own same she
should so some such than that the their theirs them then there these they this those through to too
under until up us very was we were what when where which while who whom why will with would you your
yours yourself orion juniper juniper's orion's i'm i've it's she's that's don't didn't wasn't
says said say told tells tell wrote write writes words word page records record version versions
""".split())


def stem(word: str) -> str:
    """Crude, deterministic: plural and verb endings off, then the first six letters."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    return word[:6]


def content_words(text: str) -> set[str]:
    return {stem(w) for w in _norm(text).split() if w not in _STOP and len(w) >= 3 and not w.isdigit()}


def stored_to_handles(content: str) -> tuple[str, dict[str, str]]:
    """Stored page (``[kind:id, ...]`` cites) -> the same page with ``[S1, S2]`` handles, and the
    handle -> ref map. Lets the live ``check_page`` / ``check_origin_page`` run on a stored version."""
    handle_of: dict[str, str] = {}

    def _sub(match: re.Match[str]) -> str:
        names = []
        for ref in _REF.findall(match.group(1)):
            handle_of.setdefault(ref, f"S{len(handle_of) + 1}")
            names.append(handle_of[ref])
        return "[" + ", ".join(names) + "]"

    text = _STORED_GROUP.sub(_sub, content or "")
    return text, {h: r for r, h in handle_of.items()}


def ref_kind(ref: str) -> str:
    return ref.split(":", 1)[0]


def cite_mix(refs: Iterable[str]) -> dict[str, int]:
    counts = Counter(ref_kind(r) for r in dict.fromkeys(refs))
    return {"stance": counts.get(STANCE_KIND, 0), "self_definition": counts.get(SELF_CONCEPT_KIND, 0),
            "episode_memory": counts.get(EPISODE_MEMORY_KIND, 0),
            "other": sum(v for k, v in counts.items() if k not in (STANCE_KIND, SELF_CONCEPT_KIND,
                                                                    EPISODE_MEMORY_KIND))}


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_ORD = r"(?:st|nd|rd|th)?"
_DATE = re.compile(rf"\b({'|'.join(m.capitalize() for m in _MONTHS)})\s+(\d{{1,2}}){_ORD}\b")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?:st|nd|rd|th)?(?![\w])")


def record_days(when: datetime | date) -> set[tuple[int, int]]:
    """(month, day) of a record in UTC and in Denver: an evening message in Denver is next day in UTC."""
    if isinstance(when, datetime):
        when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        return {(d.month, d.day) for d in (when.astimezone(timezone.utc).date(), when.astimezone(ZONE).date())}
    return {(when.month, when.day)}


def date_claims(sentence: str, cited_texts: Sequence[str],
                cited_dates: Iterable[datetime | date]) -> tuple[int, int]:
    """("Month D" mentions in the sentence, how many no cited record backs): a mention is backed when a
    cited record is dated that month and day (its UTC or Denver day), or its text names the same month
    and day."""
    days: set[tuple[int, int]] = set()
    for d in cited_dates:
        days |= record_days(d)
    texts = " ".join(cited_texts).lower()
    mentions = [(m.group(1), int(m.group(2))) for m in _DATE.finditer(sentence)]
    bad = 0
    for month, day in mentions:
        key = (_MONTHS.index(month.lower()) + 1, day)
        if key not in days and not re.search(rf"\b{month.lower()}\s+{day}{_ORD}\b", texts):
            bad += 1
    return len(mentions), bad


def number_claims(sentence: str, cited_texts: Sequence[str]) -> tuple[int, int]:
    """(numbers in the sentence outside "Month D" dates, how many appear in no cited record's text)."""
    def bare(n: str) -> str:
        return re.sub(r"(?:st|nd|rd|th)$", "", n)

    nums = [bare(n) for n in _NUMBER.findall(_DATE.sub(" ", sentence))]
    pool = {bare(n) for n in _NUMBER.findall(" ".join(cited_texts))}
    return len(nums), sum(1 for n in nums if n not in pool)


def sentence_support(sentence: str, cited_texts: Mapping[str, str],
                     cited_dates: Iterable[datetime | date] = (), *, cited_refs: int | None = None) -> dict[str, Any]:
    """One sentence vs the records it cites (ref -> record text, found records only). ``cited_refs``:
    how many refs the sentence cites, found or not (default: the found ones)."""
    words = content_words(sentence)
    dates, bad_dates = date_claims(sentence, list(cited_texts.values()), cited_dates)
    numbers, bad_numbers = number_claims(sentence, list(cited_texts.values()))
    per_cite = {ref: len(words & content_words(text)) for ref, text in cited_texts.items()}
    union: set[str] = set()
    for text in cited_texts.values():
        union |= content_words(text)
    shared = len(words & union)
    ratio = round(shared / len(words), 3) if words else 0.0
    if not cited_texts:
        verdict = "no_record" if (cited_refs or 0) > 0 else "uncited"
    elif not words:
        verdict = "no_content_words"
    elif shared < MIN_SHARED and len(words) >= MIN_SHARED:
        verdict = "unsupported"
    elif ratio < WEAK_RATIO:
        verdict = "weak"
    else:
        verdict = "supported"
    return {"content_words": len(words), "shared": shared, "ratio": ratio, "verdict": verdict,
            "date_claims": dates, "dates_unbacked": bad_dates, "number_claims": numbers,
            "numbers_unbacked": bad_numbers,
            "per_cite_shared": per_cite,
            "unrelated_cites": [r for r, n in per_cite.items() if n < MIN_SHARED]}


VERDICTS = ("supported", "weak", "unsupported", "no_content_words", "no_record", "uncited")
_ERROR_NAME = re.compile(r"[a-z_]+")


def error_name(error: str) -> str:
    """A check error -> its name only ("uncited: <sentence>" -> "uncited"); never page text."""
    m = _ERROR_NAME.match(error)
    return m.group(0) if m and m.end() < len(error) and error[m.end()] == ":" else "other"


def page_refs(content: str) -> list[str]:
    """Refs inside cite brackets only, in order (same parse as ``stored_to_handles``)."""
    return [r for g in _STORED_GROUP.finditer(content or "") for r in _REF.findall(g.group(1))]


def score_version(content: str, records: Mapping[str, str],
                  record_dates: Mapping[str, datetime | date] | None = None,
                  pool_loss_records: int = 0) -> dict[str, Any]:
    """One stored page version -> its loss check, cite mix and per-sentence support. ``records`` maps
    ref -> record text for every ref this eval could fetch; ``record_dates`` ref -> the record's time;
    ``pool_loss_records``: loss records in today's pool (for ``loss_dropped``)."""
    record_dates = record_dates or {}
    text, handle_to_ref = stored_to_handles(content)
    handles = {h: PageSource(ref=r, kind=ref_kind(r), text=records.get(r, ""), label="")
               for h, r in handle_to_ref.items()}
    check = check_origin_page(text, handles)
    loss_refs = memory_loss_refs(handles.values())
    sentences = []
    for sentence, refs in check.cited:
        support = sentence_support(sentence, {r: records[r] for r in dict.fromkeys(refs) if r in records},
                                   [record_dates[r] for r in refs if r in record_dates],
                                   cited_refs=len(refs))
        sentences.append({"sentence": sentence, "refs": list(dict.fromkeys(refs)),
                          "missing_records": [r for r in refs if r not in records],
                          "states_loss": bool(set(refs) & loss_refs and _LOSS_STATED.search(sentence)),
                          **support})
    verdicts = Counter(s["verdict"] for s in sentences)
    loss_stated = any(s["states_loss"] for s in sentences)
    return {
        "loss_records_cited": len(loss_refs),
        "loss_stated": loss_stated,
        "loss_dropped": (not loss_stated) and (pool_loss_records > 0 or bool(loss_refs)),
        "loss_sentences": sum(s["states_loss"] for s in sentences),
        "check_ok": check.ok,
        "check_errors": [error_name(e) for e in check.errors],
        "words": check.word_count,
        "sentences": len(sentences),
        "cite_mix": cite_mix(handle_to_ref.values()),
        "missing_records": len({r for s in sentences for r in s["missing_records"]}),
        "support": {k: verdicts.get(k, 0) for k in VERDICTS},
        "unrelated_cites": sum(len(s["unrelated_cites"]) for s in sentences),
        "cites": sum(len(s["refs"]) for s in sentences),
        "date_claims": sum(s["date_claims"] for s in sentences),
        "dates_unbacked": sum(s["dates_unbacked"] for s in sentences),
        "number_claims": sum(s["number_claims"] for s in sentences),
        "numbers_unbacked": sum(s["numbers_unbacked"] for s in sentences),
        "sentence_rows": sentences,
    }


def _words(text: str) -> list[str]:
    return _norm(text).split()


def similarity(a: str, b: str) -> float:
    return round(SequenceMatcher(None, _words(a), _words(b), autojunk=False).ratio(), 3)


def claim_churn(old: Sequence[str], new: Sequence[str], *, same: float = SAME_CLAIM) -> dict[str, Any]:
    """Sentences of two versions -> kept / dropped / added. Indices refer to the input order."""
    def best(s: str, pool: Sequence[str]) -> float:
        return max((similarity(s, p) for p in pool), default=0.0)

    dropped = [i for i, s in enumerate(old) if best(s, new) < same]
    added = [i for i, s in enumerate(new) if best(s, old) < same]
    return {"kept": len(old) - len(dropped), "dropped": dropped, "added": added}


def stability(versions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``versions``: ``{"version", "content", "refs", "sentences": [str]}`` in order -> one row per
    consecutive pair, plus first vs last."""
    pairs = list(zip(versions, versions[1:]))
    if len(versions) > 2:
        pairs.append((versions[0], versions[-1]))
    out = []
    for a, b in pairs:
        ra, rb = set(a["refs"]), set(b["refs"])
        churn = claim_churn(a["sentences"], b["sentences"])
        out.append({"from": a["version"], "to": b["version"],
                    "text_similarity": similarity(render_text(a["content"]), render_text(b["content"])),
                    "claims_kept": churn["kept"], "claims_dropped": len(churn["dropped"]),
                    "claims_added": len(churn["added"]), "dropped_idx": churn["dropped"],
                    "added_idx": churn["added"],
                    "refs_kept": len(ra & rb), "refs_dropped": len(ra - rb), "refs_added": len(rb - ra),
                    "refs_jaccard": round(len(ra & rb) / len(ra | rb), 3) if ra | rb else 1.0})
    return out


def pool_coverage(offered: Iterable[PageSource], cited_by_version: Mapping[Any, Iterable[str]]) -> dict[str, Any]:
    """Today's offered sources by kind, the loss records among them, and how many each version cites."""
    offered = list(offered)
    by_kind = Counter(s.kind for s in offered)
    loss = memory_loss_refs(offered)
    rows = {}
    for version, refs in cited_by_version.items():
        refs = set(refs)
        rows[str(version)] = {"loss_records_cited_of_offered": [len(refs & loss), len(loss)],
                              "cited_of_offered": {k: [sum(1 for s in offered if s.kind == k and s.ref in refs), n]
                                                   for k, n in by_kind.items()}}
    return {"pool_is_today": True, "offered": dict(by_kind), "loss_records_offered": len(loss),
            "by_version": rows}


def summarize_self_sense(rows: Iterable[Mapping[str, Any]], cutoff: datetime,
                         window: timedelta | None = None) -> dict[str, Any]:
    """``self_sense_eval_log`` rows (``created_at``, ``answer_source``, ``answer_text``, the two
    scores) -> before/after ``cutoff`` aggregates, per answer source. With ``window``, only rows within
    ``window`` on either side of the cutoff count."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        if window is not None and not (cutoff - window <= r["created_at"] < cutoff + window):
            continue
        when = "after" if r["created_at"] >= cutoff else "before"
        groups.setdefault(f"{when}:{r.get('answer_source') or 'unknown'}", []).append(r)

    def agg(rs: list[Mapping[str, Any]]) -> dict[str, Any]:
        n = len(rs)
        answered = [r for r in rs if str(r.get("answer_text") or "").strip()]
        return {"n": n,
                "mean_self_label": round(sum(r["self_label_score"] for r in rs) / n, 3),
                "mean_grounded_record": round(sum(r["grounded_record_score"] for r in rs) / n, 3),
                "states_memory_loss": sum(1 for r in answered if _LOSS_STATED.search(str(r["answer_text"]))),
                "answered": len(answered)}

    return {k: agg(v) for k, v in sorted(groups.items())}


# --- live, read-only ---------------------------------------------------------------------------------

_FETCH_SQL = {
    STANCE_KIND: "SELECT crystallization_id::text AS id, summary AS text, created_at AS at FROM memory_crystallizations "
                 "WHERE crystallization_id::text = ANY(%(ids)s)",
    EPISODE_MEMORY_KIND: "SELECT memory_id::text AS id, statement AS text, coalesce(occurred_at, created_at) AS at FROM episode_memory "
                         "WHERE memory_id::text = ANY(%(ids)s)",
    SELF_CONCEPT_KIND: "SELECT entry_id::text AS id, content AS text, created_at AS at FROM self_concept_history "
                       "WHERE entry_id::text = ANY(%(ids)s)",
}
_VERSIONS_SQL = ("SELECT version, created_at, content, evidence_refs FROM self_concept_history "
                 "WHERE concept_id = %(cid)s ORDER BY version, created_at")
_SELF_SENSE_SQL = ("SELECT created_at, answer_source, answer_text, self_label_score, grounded_record_score "
                   "FROM self_sense_eval_log WHERE question_key = 'what_are_you'")


def _fetch_records(conn, refs: Iterable[str]) -> tuple[dict[str, str], dict[str, datetime]]:
    by_kind: dict[str, list[str]] = {}
    for ref in dict.fromkeys(refs):
        kind, _, rid = ref.partition(":")
        by_kind.setdefault(kind, []).append(rid)
    out: dict[str, str] = {}
    dates: dict[str, datetime] = {}
    for kind, ids in by_kind.items():
        sql = _FETCH_SQL.get(kind)
        if sql:
            for row in conn.execute(sql, {"ids": ids}).fetchall():
                out[f"{kind}:{row['id']}"] = str(row["text"] or "")
                if isinstance(row.get("at"), datetime):
                    dates[f"{kind}:{row['id']}"] = row["at"]
    return out, dates


def run(conn, *, dry_run: Mapping[str, Any] | None = None, with_self_sense: bool = True,
        self_sense_days: int = 14) -> dict[str, Any]:
    from orion.memory.pages.origin import FOLD_SPEC

    versions = [dict(r) for r in conn.execute(_VERSIONS_SQL, {"cid": PAGE_ORIGIN}).fetchall()]
    if dry_run is not None:
        versions.append({"version": "dry_run", "created_at": None, "content": dry_run["stored_text"],
                         "evidence_refs": dry_run.get("refs") or []})
    all_refs = [r for v in versions for r in page_refs(v["content"])]
    records, record_dates = _fetch_records(conn, all_refs)
    rows = {name: [dict(r) for r in conn.execute(sql).fetchall()] for name, sql in FOLD_SPEC.source_sql.items()}
    offered = FOLD_SPEC.sources(rows)
    pool = pool_coverage(offered, {v["version"]: page_refs(v["content"]) for v in versions})
    scored = []
    for v in versions:
        s = score_version(v["content"], records, record_dates, pool_loss_records=pool["loss_records_offered"])
        scored.append({"version": v["version"], "created_at": str(v["created_at"] or ""),
                       "evidence_refs": len(v["evidence_refs"] or []), **s})
    stab = stability([{"version": v["version"], "content": v["content"],
                       "refs": page_refs(v["content"]),
                       "sentences": [row["sentence"] for row in s["sentence_rows"]]}
                      for v, s in zip(versions, scored)])
    report: dict[str, Any] = {"versions": scored, "stability": stab, "pool": pool}
    stored = [v for v in versions if v["created_at"] is not None]
    if with_self_sense and stored:
        ss = [dict(r) for r in conn.execute(_SELF_SENSE_SQL).fetchall()]
        cut = stored[0]["created_at"]
        report["self_sense_cutoff"] = str(cut)
        report["self_sense_window_days"] = self_sense_days
        report["self_sense"] = summarize_self_sense(ss, cut, timedelta(days=self_sense_days))
    return report


_PUBLIC_VERSION = ("version", "words", "sentences", "evidence_refs", "cite_mix", "loss_records_cited", "loss_stated",
                   "loss_dropped", "loss_sentences", "check_ok", "check_errors", "missing_records", "support",
                   "cites", "unrelated_cites", "date_claims", "dates_unbacked", "number_claims", "numbers_unbacked")
_PUBLIC_STABILITY = ("from", "to", "text_similarity", "claims_kept", "claims_dropped", "claims_added", "refs_kept",
                     "refs_dropped", "refs_added", "refs_jaccard")
_PUBLIC_SELF_SENSE = ("n", "mean_self_label", "mean_grounded_record", "states_memory_loss", "answered")


def _count_map(d: Mapping[str, Any]) -> dict[str, Any]:
    """Only keys that are plain kind/arm names and values that are numbers or number pairs."""
    def ok(v: Any) -> bool:
        return isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(x, int) for x in v))
    return {str(k): v for k, v in d.items() if re.fullmatch(r"[a-z_:]+", str(k)) and ok(v)}


def aggregates(report: Mapping[str, Any]) -> dict[str, Any]:
    """The public view, built from allowlists: counts only, no sentence, record text or id."""
    pool = report["pool"]
    return {
        "versions": [{k: v[k] for k in _PUBLIC_VERSION} for v in report["versions"]],
        "stability": [{k: s[k] for k in _PUBLIC_STABILITY} for s in report["stability"]],
        "pool": {"pool_is_today": True, "offered": _count_map(pool["offered"]),
                 "loss_records_offered": int(pool["loss_records_offered"]),
                 "by_version": {str(ver): {"loss_records_cited_of_offered": row["loss_records_cited_of_offered"],
                                           "cited_of_offered": _count_map(row["cited_of_offered"])}
                                for ver, row in pool["by_version"].items()}},
        **({"self_sense_cutoff": report["self_sense_cutoff"],
            "self_sense_window_days": report.get("self_sense_window_days"),
            "self_sense": {re.sub(r"[^a-z_:]", "", k): {f: v[f] for f in _PUBLIC_SELF_SENSE}
                           for k, v in report["self_sense"].items()}}
           if "self_sense" in report else {}),
    }


def out_path_is_safe(out: Path) -> bool:
    """False for any path inside a git checkout (this repo, a sibling worktree, any repo) or under
    /mnt/scripts: the report holds private page and record text."""
    out = out.expanduser().resolve()
    if Path("/mnt/scripts") in out.parents or out == Path("/mnt/scripts"):
        return False
    probe = out.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        hit = subprocess.run(["git", "-C", str(probe), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return True
    return hit.returncode != 0


def write_private(out: Path, text: str) -> None:
    """Owner-only from creation: the directory 0700, the file 0600 (no window under the umask)."""
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.chmod(out, 0o600)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--page-json", help="optional dry-run JSON (scripts/core_page_dry_run.py --page origin --out)")
    ap.add_argument("--out", help="full per-sentence report, LOCAL path (default ~/.orion/evals/page_origin_history_<utc>.json)")
    ap.add_argument("--no-self-sense", action="store_true")
    ap.add_argument("--self-sense-days", type=int, default=14, help="window each side of the first page:origin")
    args = ap.parse_args(argv)
    out = Path(args.out).expanduser() if args.out else (
        DEFAULT_OUT_DIR / f"page_origin_history_{datetime.now(timezone.utc):%Y-%m-%dT%H%MZ}.json")
    if not out_path_is_safe(out):
        raise SystemExit("--out must be outside any git checkout and /mnt/scripts: the report holds private text")
    dry = json.loads(Path(args.page_json).expanduser().read_text()) if args.page_json else None
    if dry is not None and not str(dry.get("stored_text") or "").strip():
        raise SystemExit("--page-json has no stored_text (use scripts/core_page_dry_run.py --page origin --out)")
    sys.path.insert(0, str(REPO))
    import importlib.util

    import psycopg
    from psycopg.rows import dict_row

    spec = importlib.util.spec_from_file_location("core_page_dry_run", REPO / "scripts" / "core_page_dry_run.py")
    dry_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dry_mod)
    with psycopg.connect(dry_mod.default_dsn(), row_factory=dict_row) as conn:
        conn.read_only = True
        report = run(conn, dry_run=dry, with_self_sense=not args.no_self_sense,
                     self_sense_days=args.self_sense_days)
    write_private(out, json.dumps(report, indent=2, default=str))
    print(json.dumps(aggregates(report), indent=2, default=str))
    print(f"full report (private, local only): {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
