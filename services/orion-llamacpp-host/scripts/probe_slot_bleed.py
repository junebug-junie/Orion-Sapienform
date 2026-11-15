#!/usr/bin/env python3
"""Cross-conversation bleed canary for a multi-slot llama.cpp worker (llama.cpp #27148).

GPU pool stage 7.2 (docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md, D2 and the
"#27148 probe" section). The 2026-10-01 probe did not reproduce on metacog/fast (dense Qwen3-8B,
4 x 4K slots, prompts <= 3.2K tokens, no tool turns) and the spec says that does not clear the
agent lanes. This runs the same detectors against the LIVE agent-gpu2 worker once it serves
Ternary-Bonsai-2-27B with 2 x 131K slots, with prompts >= 4.5K tokens and tool-call turns.

Leased (leased-evals design, docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md):
the probe takes TWO operator holds on the lane (default agent-gpu2, class operator_agent_gpu2,
background priority) -- one per llama.cpp slot, so the simultaneous phases really run side by side
and none of Orion's work shares the seat while they run. Every model call goes through
orion-llm-gateway under one of the holds (orion/evals/leased_llm.py); request fields (id_slot,
cache_prompt) and the reply's timings pass through unchanged. /props, /slots, /apply-template and
/tokenize are read from the URL the pool granted (they run no model). When the pool takes a hold
back (an owner waits), the run stops INCONCLUSIVE. There is no direct-URL mode.

What it does (read-only toward server config; synthetic prompts only):
  - Refuses to start unless /props shows the expected model and >= 2 slots, /slots shows every
    slot idle, AND the GPU pool (athena, --pool-url) shows no granted or recalling lease on the
    role other than the probe's own holds and their calls. Re-checks both before and after every
    phase; work that overlapped a phase makes the run INCONCLUSIVE and stops it.
  - Every conversation carries its own random codeword (canary). Phases: seed, sequential
    unrelated (the sequential repro), id_slot-pinned pairs (stale live slot), simultaneous pairs
    (the original repro), simultaneous tool-call conversations (assistant tool_call + tool result
    carrying the canary), and a shared-prefix pair (long common prefix, private tails).
  - Detectors: (1) a FOREIGN canary in any reply text, reasoning, or tool-call arguments;
    (2) cache_n above the TRUE longest common prefix (in tokens) between this prompt and any other
    prompt the probe sent, computed with the worker's own /apply-template + /tokenize. That covers
    the shared-prefix pair too, where #27148's RAM-cache restore is most likely to run: a stale KV
    tail the model happens not to echo still shows up as cache_n past the real common prefix.

Verdicts (exit code): PASS 0, LEAK 1, REFUSED 2 (precondition), INCONCLUSIVE 3 (any failed or
missing request, real work overlapping a phase, a prompt whose common prefix could not be
computed, prompts under --min-prompt-tokens, or the model never repeating its own codeword -- a
detector that cannot see its own canary cannot see a foreign one either). INCONCLUSIVE is never a pass.
Evidence is written owner-only (0700 dir, 0600 files): a real leak would put private text in it.

Run from athena (nothing for Juniper to run):
  ORION_BUS_URL=redis://100.92.216.81:6379/0 python3 services/orion-llamacpp-host/scripts/probe_slot_bleed.py
Writes <out>/requests.jsonl and <out>/summary.json (default /tmp/slot-bleed-probe/<utc>/).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
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
# llama.cpp may re-evaluate a few tokens at the prompt boundary; reuse past LCP + this is a leak.
LCP_SLACK = 8
ACTIVE_LEASE = frozenset({"granted", "recalling"})
TOOL = {"type": "function", "function": {
    "name": "lookup_record", "description": "Fetch an archived record by id.",
    "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}}}


def canary(rng: random.Random) -> str:
    return "CW-" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(8))


def filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


HOLDER = "operator:probe-slot-bleed"
REPO = Path(__file__).resolve().parents[3]


@dataclass
class Probe:
    url: str                       # the granted worker URL (display + evidence only)
    out: Path
    clients: list[Any] = field(default_factory=list)   # one httpx-shaped client per hold (leased_llm.LaneClient)
    own_leases: frozenset[str] = frozenset()            # the probe's holds: never "real work"
    pool_url: str = "http://127.0.0.1:8127"
    role: str = "agent-gpu2"
    expect_model: str = "Ternary-Bonsai"
    min_prompt_tokens: int = 4500
    words: int = 5000
    max_tokens: int = 96
    timeout_sec: float = 600.0
    idle_wait_sec: float = 300.0
    pace_sec: float = 2.0
    seed: int | None = None
    rng: random.Random = field(init=False)
    canaries: dict[str, str] = field(default_factory=dict)   # canary -> conversation id
    records: list[dict[str, Any]] = field(default_factory=list)
    expected: int = 0                                        # requests dispatched by phases
    tainted: str | None = None                               # why real work overlapped the probe
    recalled: str | None = None                              # the pool took a hold back
    _prompts: list[tuple[str, list[int]]] = field(default_factory=list)   # (conv, tokens) sent so far
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _n: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed) if self.seed is not None else random.SystemRandom()

    # --- HTTP -------------------------------------------------------------------------------
    def _pool(self) -> Any:
        with urllib.request.urlopen(self.pool_url.rstrip("/") + "/v1/pool", timeout=30) as resp:
            return json.load(resp)

    def _client(self, n: int = 0) -> Any:
        if not self.clients:
            raise RuntimeError("no operator hold: the probe calls the worker only under a pool hold")
        return self.clients[n % len(self.clients)]

    def _get(self, path: str) -> Any:
        r = self._client().get(path, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict[str, Any], timeout: float | None = None, hold: int = 0) -> dict[str, Any]:
        r = self._client(hold).post(path, json=body, timeout=timeout or self.timeout_sec)
        r.raise_for_status()
        return r.json()

    # --- evidence (owner-only) -----------------------------------------------------------------
    def _mkout(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.out, 0o700)

    def _append(self, name: str, text: str, mode: str = "a") -> None:
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if mode == "a" else os.O_TRUNC)
        fd = os.open(self.out / name, flags, 0o600)
        with os.fdopen(fd, mode, encoding="utf-8") as fh:
            fh.write(text)

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
        busy = self.wait_quiet()
        return None if busy is None else f"seat busy with real work: {busy}"

    def busy_reason(self) -> str | None:
        """None when the seat is quiet: every llama.cpp slot idle and no active pool lease on the role.
        Any failure to read either counts as busy -- quiet must be proven, not assumed."""
        try:
            slots = self._get("/slots")
        except Exception as exc:  # noqa: BLE001
            return f"/slots unreadable: {exc!r}"
        if not (isinstance(slots, list) and slots and all(
                isinstance(s, dict) and s.get("is_processing") is False for s in slots)):
            return "a llama.cpp slot is processing"
        try:
            state = self._pool()
        except Exception as exc:  # noqa: BLE001
            return f"pool state unreadable at {self.pool_url}: {exc!r}"
        active = [l.get("lease_id") for l in state.get("leases") or []
                  if l.get("role") == self.role and l.get("status") in ACTIVE_LEASE
                  and l.get("lease_id") not in self.own_leases and l.get("hold_lease_id") not in self.own_leases]
        return f"pool leases active on {self.role}: {active}" if active else None

    def wait_quiet(self) -> str | None:
        deadline = time.monotonic() + self.idle_wait_sec
        while True:
            why = self.busy_reason()
            if why is None or time.monotonic() >= deadline:
                return why
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

    def _body(self, messages: list[dict[str, Any]], tools: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
        if tools:
            body["tools"] = [TOOL]
        return body

    def register(self, cid: str, messages: list[dict[str, Any]], tools: bool = False) -> list[int] | None:
        """Render + tokenize the prompt exactly as the worker will, and remember it for LCP checks.
        Must run before the prompt is sent (together() registers a pair before either goes out)."""
        try:
            prompt = self._post("/apply-template", self._body(messages, tools), timeout=60)["prompt"]
            tokens = [int(t) for t in self._post("/tokenize", {"content": prompt}, timeout=60)["tokens"]]
        except Exception:  # noqa: BLE001
            return None
        with self._lock:
            self._prompts.append((cid, tokens))
        return tokens

    def true_lcp(self, cid: str, tokens: list[int]) -> int:
        best = 0
        with self._lock:
            others = [t for c, t in self._prompts if c != cid]
        for other in others:
            n = 0
            for a, b in zip(tokens, other):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return best

    def send(self, phase: str, cid: str, own: str, messages: list[dict[str, Any]], *,
             id_slot: int | None = None, tools: bool = False, tokens: list[int] | None = None,
             registered: bool = False, hold: int = 0) -> dict[str, Any]:
        t0 = time.time()
        rec: dict[str, Any] = {"t": t0, "phase": phase, "conv": cid, "own_canary": own, "id_slot": id_slot,
                               "tools": tools, "error": None, "reply": "", "own_found": False,
                               "foreign_canaries": [], "foreign_convs": [], "cache_n": None, "prompt_n": None,
                               "prompt_tokens": None, "true_lcp": None, "cache_over_lcp": False}
        try:   # a crashed request is recorded as an error, never silently dropped
            if not registered:
                tokens = self.register(cid, messages, tools)
            body = {**self._body(messages, tools), "max_tokens": self.max_tokens, "temperature": 0,
                    "cache_prompt": True}
            if id_slot is not None:
                body["id_slot"] = id_slot
            resp = self._post("/v1/chat/completions", body, hold=hold)
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            seen = " ".join([str(msg.get("content") or ""), str(msg.get("reasoning_content") or ""),
                             *(json.dumps(tc) for tc in msg.get("tool_calls") or [])])
            timings = resp.get("timings") or {}
            cache_n, prompt_n = timings.get("cache_n"), timings.get("prompt_n")
            foreign = sorted(c for c in self.canaries if c != own and c in seen)
            lcp = self.true_lcp(cid, tokens) if tokens is not None else None
            rec.update(reply=seen[:2000], own_found=own in seen, foreign_canaries=foreign,
                       foreign_convs=sorted({self.canaries[c] for c in foreign}), cache_n=cache_n,
                       prompt_n=prompt_n, true_lcp=lcp,
                       prompt_tokens=(cache_n or 0) + prompt_n if prompt_n is not None else None,
                       cache_over_lcp=bool(cache_n is not None and lcp is not None and cache_n > lcp + LCP_SLACK))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)
            if type(exc).__name__ == "LaneRecalled":   # leased_llm: the pool took a hold back
                self.recalled = str(exc)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        with self._lock:
            self.records.append(rec)
            self._append("requests.jsonl", json.dumps(rec) + "\n")
        flag = ("LEAK" if rec["foreign_canaries"] else "CACHE>LCP" if rec["cache_over_lcp"]
                else "ERR" if rec["error"] else "ok")
        print(f"{phase:14} {cid:8} slot={id_slot} cache_n={rec['cache_n']} lcp={rec['true_lcp']} "
              f"prompt_n={rec['prompt_n']} own={rec['own_found']} {flag}", flush=True)
        return rec

    def one(self, phase: str, conv: tuple[str, str, list[dict[str, Any]]], **kw) -> None:
        self.expected += 1
        self.send(phase, *conv, **kw)
        time.sleep(self.pace_sec)

    def together(self, phase: str, convs: list[tuple[str, str, list[dict[str, Any]]]], tools: bool = False) -> None:
        self.expected += len(convs)
        pre = [self.register(cid, m, tools) for cid, _c, m in convs]
        threads = [threading.Thread(target=self.send, args=(phase, cid, c, m),
                                    kwargs={"tools": tools, "tokens": toks, "registered": True, "hold": i})
                   for i, ((cid, c, m), toks) in enumerate(zip(convs, pre))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # --- phases -------------------------------------------------------------------------------
    def run(self, rounds: int = 2) -> dict[str, Any]:
        self._mkout()
        refused = self.precheck()
        if refused:
            return self.summarize(refused=refused)
        phases = [self._seed, self._sequential, self._pinned, self._simultaneous, self._tool_turns,
                  self._shared_prefix]
        for _ in range(rounds):
            for phase in phases:
                busy = self.wait_quiet()
                if busy is not None:
                    self.tainted = f"before {phase.__name__.lstrip('_')}: {busy}"
                    return self.summarize()
                phase()
                if self.recalled:
                    self.tainted = f"the GPU pool took a hold back during {phase.__name__.lstrip('_')}: {self.recalled}"
                    return self.summarize()
                busy = self.busy_reason()
                if busy is not None:
                    self.tainted = f"during {phase.__name__.lstrip('_')}: {busy}"
                    return self.summarize()
                time.sleep(self.pace_sec)
        return self.summarize()

    def _seed(self) -> None:
        for _ in range(2):
            self.one("seed", self.conv("S"))

    def _sequential(self) -> None:
        for _ in range(3):
            self.one("sequential", self.conv("Q"))

    def _pinned(self) -> None:
        for slot in (0, 1):
            self.one("pinned_A", self.conv("PA", self.words + 800), id_slot=slot)
            self.one("pinned_B", self.conv("PB"), id_slot=slot)

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
            self.one("shared_prefix", (cid, c, [{"role": "system", "content": doc},
                                                {"role": "user", "content": ASK}]))

    # --- verdict ------------------------------------------------------------------------------
    def summarize(self, refused: str | None = None) -> dict[str, Any]:
        recs = list(self.records)
        ok = [r for r in recs if not r["error"]]
        sized = [r for r in ok if r["prompt_tokens"] is not None]
        short = [r for r in sized if r["prompt_tokens"] < self.min_prompt_tokens]
        unchecked = [r for r in ok if r["true_lcp"] is None or r["cache_n"] is None]
        leaks = [r for r in recs if r["foreign_canaries"]]
        cache_anomalies = [r for r in recs if r["cache_over_lcp"]]
        own_rate = (sum(r["own_found"] for r in ok) / len(ok)) if ok else 0.0
        if refused:
            verdict, why = "REFUSED", refused
        elif leaks or cache_anomalies:   # a leak is a leak even in a cut-short run
            verdict, why = "LEAK", f"{len(leaks)} foreign-canary replies, {len(cache_anomalies)} cache_n past the true common prefix"
        elif self.tainted:
            verdict, why = "INCONCLUSIVE", f"real work overlapped the probe ({self.tainted})"
        elif not recs or len(recs) != self.expected or len(ok) != len(recs):
            verdict, why = "INCONCLUSIVE", (f"{len(recs) - len(ok)} failed and {self.expected - len(recs)} missing "
                                            f"of {self.expected} requests")
        elif unchecked:
            verdict, why = "INCONCLUSIVE", f"{len(unchecked)} requests without a computable common prefix or cache_n"
        elif len(sized) < len(ok) or short:
            verdict, why = "INCONCLUSIVE", (f"{len(short)} prompts under {self.min_prompt_tokens} tokens, "
                                            f"{len(ok) - len(sized)} without timings")
        elif own_rate < 0.5:
            verdict, why = "INCONCLUSIVE", f"model repeated its own codeword in only {own_rate:.0%} of replies"
        else:
            verdict, why = "PASS", "no foreign canary, no cache reuse past the true common prefix"
        summary = {
            "verdict": verdict, "why": why, "url": self.url, "pool_url": self.pool_url,
            "holds": sorted(self.own_leases),
            "requests": len(recs), "expected": self.expected, "ok": len(ok),
            "leaks": [{k: r[k] for k in ("phase", "conv", "foreign_convs", "id_slot")} for r in leaks],
            "cache_over_lcp": [{k: r[k] for k in ("phase", "conv", "cache_n", "true_lcp", "prompt_n")}
                               for r in cache_anomalies],
            "own_codeword_rate": round(own_rate, 3),
            "prompt_tokens_min": min((r["prompt_tokens"] for r in sized), default=None),
            "prompt_tokens_max": max((r["prompt_tokens"] for r in sized), default=None),
            "phases": sorted({r["phase"] for r in recs}),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        self._mkout()
        self._append("summary.json", json.dumps(summary, indent=1), mode="w")
        print(f"VERDICT: {verdict} -- {why} (evidence: {self.out})", flush=True)
        return summary


EXIT = {"PASS": 0, "LEAK": 1, "REFUSED": 2, "INCONCLUSIVE": 3}


@contextlib.contextmanager
def operator_holds(lane: str, n: int):
    """``n`` operator holds on ``lane`` (one per slot), released on every exit."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from orion.evals.leased_llm import operator_lane_sync

    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(operator_lane_sync(lane, holder=HOLDER)) for _ in range(n)]


def main(argv: list[str] | None = None, holds=operator_holds) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--lane", default="agent-gpu2", help="operator lane to hold (leased_llm.OPERATOR_LANES)")
    ap.add_argument("--holds", type=int, default=2, help="operator holds to take (one per slot)")
    ap.add_argument("--pool-url", default="http://127.0.0.1:8127",
                    help="GPU pool HTTP (athena); the seat must show no active lease but the probe's own")
    ap.add_argument("--role", default="agent-gpu2", help="pool role the worker serves")
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
    try:
        with holds(args.lane, max(1, args.holds)) as lanes:
            probe = Probe(url=lanes[0].worker_url, out=out, clients=[l.http_client() for l in lanes],
                          own_leases=frozenset(l.lease_id for l in lanes), pool_url=args.pool_url,
                          role=args.role, expect_model=args.expect_model,
                          min_prompt_tokens=args.min_prompt_tokens, words=args.words,
                          idle_wait_sec=args.idle_wait_sec, pace_sec=args.pace_sec, seed=args.seed)
            return EXIT[probe.run(rounds=args.rounds)["verdict"]]
    except Exception as exc:  # noqa: BLE001 -- LaneUnavailable (no seat) or a recall at release
        if type(exc).__name__ == "LaneRecalled":
            print(f"INCONCLUSIVE: {exc}", file=sys.stderr)
            return EXIT["INCONCLUSIVE"]
        if type(exc).__name__ == "LaneUnavailable":
            print(f"REFUSED: {exc}", file=sys.stderr)
            return EXIT["REFUSED"]
        raise


if __name__ == "__main__":
    sys.exit(main())
