#!/usr/bin/env python3
"""Cross-conversation bleed canary for a multi-slot llama.cpp worker (llama.cpp #27148).

GPU pool stage 7.2 (docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md, D2 and the
"#27148 probe" section). The 2026-10-01 probe did not reproduce on metacog/fast (dense Qwen3-8B,
4 x 4K slots, prompts <= 3.2K tokens, no tool turns) and the spec says that does not clear the
agent lanes. This runs the same detectors against the LIVE agent-gpu2 worker once it serves
Ternary-Bonsai-2-27B with 2 x 131K slots, with prompts >= 4.5K tokens and tool-call turns.

What it does (read-only toward server config; synthetic prompts only):
  - Refuses to start unless /props shows the expected model and >= 2 slots, and /slots shows every
    slot idle. Re-checks idleness before each phase and stops if real work arrives: these calls
    go straight to the worker (athena is allowed through circe's port gate), not through a pool
    lease, so they must not compete with Orion's own runs.
  - Every conversation carries its own random codeword (canary). Phases: seed, sequential
    unrelated (the sequential repro), id_slot-pinned pairs (stale live slot), simultaneous pairs
    (the original repro), simultaneous tool-call conversations (assistant tool_call + tool result
    carrying the canary), and a shared-prefix pair (long common prefix, private tails).
  - Detectors: (1) a FOREIGN canary in any reply text, reasoning, or tool-call arguments;
    (2) cache_n above the true common prefix for unrelated conversations (each prompt starts with
    its own canary, so honest reuse is only the chat-template header).

Verdicts (exit code): PASS 0, LEAK 1, REFUSED 2 (precondition), INCONCLUSIVE 3 (too many errors,
prompts under --min-prompt-tokens, or the model never repeats its own codeword -- a detector that
cannot see its own canary cannot see a foreign one either). INCONCLUSIVE is never a pass.

Run from athena after the 7.2 deploy (nothing for Juniper to run):
  python3 services/orion-llamacpp-host/scripts/probe_slot_bleed.py --url http://100.112.254.99:8016
Writes <out>/requests.jsonl and <out>/summary.json (default /tmp/slot-bleed-probe/<utc>/).
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lantern meadow nickel "
         "orchid pewter quartz river saffron timber umber violet willow xenon yarrow zephyr anchor "
         "bramble cobalt dune estuary falcon glacier heron indigo jasper kestrel lichen marble nectar "
         "obsidian prairie quill ridge sorrel thistle tundra vellum walnut").split()
ASK = ("List every codeword of the form CW-XXXXXXXX that appears anywhere in this conversation, "
       "one per line, nothing else.")
TOOLS_ALLOWANCE = 512
TOOL = {"type": "function", "function": {
    "name": "lookup_record", "description": "Fetch an archived record by id.",
    "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}}}


def canary(rng: random.Random) -> str:
    return "CW-" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


@dataclass
class Probe:
    url: str
    out: Path
    expect_model: str = "Ternary-Bonsai"
    min_prompt_tokens: int = 4500
    words: int = 5000
    header_allowance: int = 64
    max_tokens: int = 96
    timeout_sec: float = 600.0
    idle_wait_sec: float = 300.0
    pace_sec: float = 2.0
    seed: int | None = None
    rng: random.Random = field(init=False)
    canaries: dict[str, str] = field(default_factory=dict)   # canary -> conversation id
    records: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _n: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed) if self.seed is not None else random.SystemRandom()

    # --- HTTP -------------------------------------------------------------------------------
    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.url + path, timeout=30) as resp:
            return json.load(resp)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            return json.load(resp)

    # --- preconditions ------------------------------------------------------------------------
    def precheck(self) -> str | None:
        """None when the worker is the expected multi-slot model, else why the probe refuses."""
        try:
            props = self._get("/props")
        except Exception as exc:  # noqa: BLE001
            return f"/props unreachable: {exc!r}"
        model = str(props.get("model_path") or "")
        slots = int(props.get("total_slots") or 0)
        if self.expect_model and self.expect_model not in model:
            return f"model_path {model!r} does not contain {self.expect_model!r}"
        if slots < 2:
            return f"total_slots={slots}: the bleed needs >= 2 slots"
        return None if self.wait_idle() else "slots busy with real work"

    def idle(self) -> bool:
        slots = self._get("/slots")
        return isinstance(slots, list) and bool(slots) and all(
            isinstance(s, dict) and s.get("is_processing") is False for s in slots)

    def wait_idle(self) -> bool:
        deadline = time.monotonic() + self.idle_wait_sec
        while True:
            try:
                if self.idle():
                    return True
            except Exception:  # noqa: BLE001
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(10.0, self.idle_wait_sec))

    # --- conversations ------------------------------------------------------------------------
    def _cid(self, tag: str) -> str:
        with self._lock:
            self._n += 1
            return f"{tag}{self._n}"

    def conv(self, tag: str, words: int | None = None) -> tuple[str, str, list[dict[str, Any]]]:
        cid, c = self._cid(tag), canary(self.rng)
        self.canaries[c] = cid
        doc = (f"{c} is the only codeword for this document. Document {cid}: {filler(self.rng, words or self.words)} "
               f"End of document. Codeword again: {c}.")
        return cid, c, [{"role": "system", "content": doc}, {"role": "user", "content": ASK}]

    def tool_conv(self, tag: str) -> tuple[str, str, list[dict[str, Any]]]:
        cid, c = self._cid(tag), canary(self.rng)
        self.canaries[c] = cid
        call_id = "call_" + "".join(self.rng.choice(string.ascii_lowercase) for _ in range(10))
        doc = f"{c} opens case file {cid}. Background: {filler(self.rng, self.words)}"
        return cid, c, [
            {"role": "system", "content": doc},
            {"role": "user", "content": f"Fetch record {cid} and tell me its codeword."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
                "name": "lookup_record", "arguments": json.dumps({"record_id": cid})}}]},
            {"role": "tool", "tool_call_id": call_id,
             "content": f"Record {cid}: codeword {c}. Notes: {filler(self.rng, 300)}"},
            {"role": "user", "content": ASK},
        ]

    def send(self, phase: str, cid: str, own: str, messages: list[dict[str, Any]], *,
             unrelated: bool = True, id_slot: int | None = None, tools: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "max_tokens": self.max_tokens, "temperature": 0,
                                "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
        if id_slot is not None:
            body["id_slot"] = id_slot
        if tools:
            body["tools"] = [TOOL]
        t0 = time.time()
        try:
            resp, err = self._post(body), None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            resp, err = {}, repr(exc)
        msg = ((resp.get("choices") or [{}])[0].get("message") or {}) if resp else {}
        seen = " ".join([str(msg.get("content") or ""), str(msg.get("reasoning_content") or ""),
                         *(json.dumps(tc) for tc in msg.get("tool_calls") or [])])
        timings = resp.get("timings") or {}
        cache_n, prompt_n = timings.get("cache_n"), timings.get("prompt_n")
        foreign = sorted(c for c in self.canaries if c != own and c in seen)
        # Honest reuse between unrelated conversations is the template header, plus the identical
        # tools block when the template renders it ahead of the system text (template-dependent).
        allowance = self.header_allowance + (TOOLS_ALLOWANCE if tools else 0)
        rec = {
            "t": t0, "elapsed_s": round(time.time() - t0, 2), "phase": phase, "conv": cid, "own_canary": own,
            "id_slot": id_slot, "tools": tools, "unrelated": unrelated, "error": err,
            "reply": seen[:2000], "own_found": own in seen, "foreign_canaries": foreign,
            "foreign_convs": sorted({self.canaries[c] for c in foreign}),
            "cache_n": cache_n, "prompt_n": prompt_n,
            "prompt_tokens": (cache_n or 0) + (prompt_n or 0) if prompt_n is not None else None,
            "cache_over_header": bool(unrelated and cache_n is not None and cache_n > allowance),
        }
        with self._lock:
            self.records.append(rec)
            with open(self.out / "requests.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        flag = "LEAK" if foreign else ("CACHE>HDR" if rec["cache_over_header"] else ("ERR" if err else "ok"))
        print(f"{phase:14} {cid:8} slot={id_slot} cache_n={cache_n} prompt_n={prompt_n} "
              f"own={rec['own_found']} {flag}", flush=True)
        return rec

    def together(self, phase: str, convs: list[tuple[str, str, list[dict[str, Any]]]], **kw) -> None:
        threads = [threading.Thread(target=self.send, args=(phase, cid, c, m), kwargs=kw) for cid, c, m in convs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # --- phases -------------------------------------------------------------------------------
    def run(self, rounds: int = 2) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        refused = self.precheck()
        if refused:
            return self.summarize(refused=refused)
        phases = [self._seed, self._sequential, self._pinned, self._simultaneous, self._tool_turns,
                  self._shared_prefix]
        for _ in range(rounds):
            for phase in phases:
                if not self.wait_idle():
                    return self.summarize(refused="real work arrived mid-probe; stopped")
                phase()
                time.sleep(self.pace_sec)
        return self.summarize()

    def _seed(self) -> None:
        for _ in range(2):
            self.send("seed", *self.conv("S"))
            time.sleep(self.pace_sec)

    def _sequential(self) -> None:
        for _ in range(3):
            self.send("sequential", *self.conv("Q"))
            time.sleep(self.pace_sec)

    def _pinned(self) -> None:
        for slot in (0, 1):
            self.send("pinned_A", *self.conv("PA", self.words + 800), id_slot=slot)
            time.sleep(self.pace_sec)
            self.send("pinned_B", *self.conv("PB"), id_slot=slot)
            time.sleep(self.pace_sec)

    def _simultaneous(self) -> None:
        self.together("simultaneous", [self.conv("C"), self.conv("C")])

    def _tool_turns(self) -> None:
        self.together("tool_turn", [self.tool_conv("T"), self.tool_conv("T")], tools=True)

    def _shared_prefix(self) -> None:
        prefix = filler(self.rng, self.words)
        for i, tail in enumerate((600, 60)):   # X longer than Y: a stale tail sits past the true LCP
            cid, c = self._cid("SP" + "XY"[i]), canary(self.rng)
            self.canaries[c] = cid
            doc = f"Shared archive. {prefix} Private appendix for {cid}: the codeword is {c}. {filler(self.rng, tail)}"
            self.send("shared_prefix", cid, c, [{"role": "system", "content": doc}, {"role": "user", "content": ASK}],
                      unrelated=False)
            time.sleep(self.pace_sec)

    # --- verdict ------------------------------------------------------------------------------
    def summarize(self, refused: str | None = None) -> dict[str, Any]:
        recs = self.records
        ok = [r for r in recs if not r["error"]]
        sized = [r for r in ok if r["prompt_tokens"] is not None]
        short = [r for r in sized if r["prompt_tokens"] < self.min_prompt_tokens]
        leaks = [r for r in recs if r["foreign_canaries"]]
        cache_anomalies = [r for r in recs if r["cache_over_header"]]
        own_rate = (sum(r["own_found"] for r in ok) / len(ok)) if ok else 0.0
        if refused:
            verdict, why = "REFUSED", refused
        elif leaks or cache_anomalies:
            verdict, why = "LEAK", f"{len(leaks)} foreign-canary replies, {len(cache_anomalies)} cache_n over header"
        elif not recs or len(ok) < 0.9 * len(recs):
            verdict, why = "INCONCLUSIVE", f"{len(recs) - len(ok)} of {len(recs)} requests failed"
        elif len(sized) < len(ok) or short:
            verdict, why = "INCONCLUSIVE", (f"{len(short)} prompts under {self.min_prompt_tokens} tokens, "
                                            f"{len(ok) - len(sized)} without timings")
        elif own_rate < 0.5:
            verdict, why = "INCONCLUSIVE", f"model repeated its own codeword in only {own_rate:.0%} of replies"
        else:
            verdict, why = "PASS", "no foreign canary, no cache reuse past the template header"
        summary = {
            "verdict": verdict, "why": why, "url": self.url, "requests": len(recs), "ok": len(ok),
            "leaks": [{k: r[k] for k in ("phase", "conv", "foreign_convs", "id_slot")} for r in leaks],
            "cache_over_header": [{k: r[k] for k in ("phase", "conv", "cache_n", "prompt_n")} for r in cache_anomalies],
            "own_codeword_rate": round(own_rate, 3),
            "prompt_tokens_min": min((r["prompt_tokens"] for r in sized), default=None),
            "prompt_tokens_max": max((r["prompt_tokens"] for r in sized), default=None),
            "phases": sorted({r["phase"] for r in recs}),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"VERDICT: {verdict} -- {why} (evidence: {self.out})", flush=True)
        return summary


EXIT = {"PASS": 0, "LEAK": 1, "REFUSED": 2, "INCONCLUSIVE": 3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--url", default="http://100.112.254.99:8016", help="worker base URL (agent-gpu2)")
    ap.add_argument("--out", default=None, help="evidence dir (default /tmp/slot-bleed-probe/<utc>)")
    ap.add_argument("--expect-model", default="Ternary-Bonsai", help="substring /props model_path must contain")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--words", type=int, default=5000, help="filler words per document (~1.3 tokens each)")
    ap.add_argument("--min-prompt-tokens", type=int, default=4500)
    ap.add_argument("--idle-wait-sec", type=float, default=300.0)
    ap.add_argument("--pace-sec", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    out = Path(args.out or f"/tmp/slot-bleed-probe/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    probe = Probe(url=args.url.rstrip("/"), out=out, expect_model=args.expect_model,
                  min_prompt_tokens=args.min_prompt_tokens, words=args.words,
                  idle_wait_sec=args.idle_wait_sec, pace_sec=args.pace_sec, seed=args.seed)
    return EXIT[probe.run(rounds=args.rounds)["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
