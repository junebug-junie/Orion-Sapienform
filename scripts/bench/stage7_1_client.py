#!/usr/bin/env python3
"""GPU pool stage 7.1 bake-off client: depth bench + llama.cpp #27148 bleed canary. Stdlib only.

Spec: docs/superpowers/specs/2026-09-30-gpu-pool-stage7-concurrency.md (acceptance checks 1-2).
Driven by scripts/bench/stage7_1_bakeoff.sh on circe; runnable alone against any llama.cpp server:

    stage7_1_client.py preflight-check --pool-json pool.json --gpu2-used-mib 1248
    stage7_1_client.py bench   --url http://127.0.0.1:8017 --out bench.json --depths 14000,32000,61000,100000
    stage7_1_client.py canary  --url http://127.0.0.1:8017 --out canary.json
    stage7_1_client.py summarize --results-dir DIR

Bench (check 1): per depth, 1 run alone and 2 runs at once. Prefill is timed on a cold prompt; decode
is timed on a second, warm request of the same prompt (its prefix is in the slot), so two runs really
decode together at that depth instead of one decoding while the other still prefills.
Pass: total decode tok/s of 2 runs >= 1.3x the tok/s of 1 run, at every depth.

Canary (check 2): conversations that each carry their own random nonces (a document codeword and
per-tool-call ledger values). Two shapes: simultaneous pairs, and 4 conversations interleaved over
the server's slots (every turn lands on a slot last used by another conversation, so LRU reuse and
the idle-slot RAM cache restore run constantly). Plus fresh single-turn conversations between
rounds (the upstream symptom: an unrelated finished conversation restored into a fresh slot).
Two detectors, as in the 2026-10-01 metacog/fast probe:
  1. a nonce owned by another conversation appears in content, reasoning, or tool-call arguments;
  2. the server's reused-token count (timings.cache_n) exceeds what any previously sent prompt
     really shares with this one (token LCP via /apply-template + /tokenize, plus the previous
     reply's length when that prompt is a full prefix of this one).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PASS_RATIO = 1.3
SPEC_DEPTHS = (14_000, 32_000, 61_000, 100_000)
NONCE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I: survives a model retyping it
NONCE_PREFIX = "NX"
NONCE_LEN = 10
LIVE_LEASE = frozenset({"granted", "recalling"})
WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lantern meadow nickel "
         "orchid pewter quartz river saffron timber umber violet willow xenon yarrow zephyr anchor "
         "bramble cobalt dune estuary falcon glacier heron indigo jasper kestrel lichen marble nectar "
         "obsidian prairie quill ridge sorrel thistle tundra vellum walnut").split()
TOOLS = [{"type": "function", "function": {
    "name": "lookup_ledger", "description": "Look up one value in this conversation's private ledger.",
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]


# --------------------------------------------------------------------------- pure helpers (tested)

def new_nonce(rng: random.Random) -> str:
    return NONCE_PREFIX + "".join(rng.choice(NONCE_ALPHABET) for _ in range(NONCE_LEN))


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def nonces_in(text: str, registry: dict[str, str]) -> set[str]:
    """Registered nonces present in ``text``. Matched after dropping case, spaces and punctuation, so
    a model that writes 'nx-abcd efgh 23' still counts. 32^10 random suffixes: no chance matches."""
    flat = _norm(text)
    return {n for n in registry if n in flat}


def foreign_nonces(text: str, own_conv: str, registry: dict[str, str]) -> list[dict[str, str]]:
    """Nonces in ``text`` that belong to a conversation other than ``own_conv``: a bleed."""
    return [{"nonce": n, "owner": registry[n]} for n in sorted(nonces_in(text, registry))
            if registry[n] != own_conv]


def lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def legit_reuse_bound(tokens: list[int], history: list[tuple[list[int], int]], floor: int = 0) -> int:
    """Most prompt tokens the server could honestly reuse for ``tokens``: the longest common prefix
    with any prompt sent before, plus that request's generated length when its whole prompt is a
    prefix of this one (the slot also holds what it generated, which may match the re-rendered
    reply). ``history`` = [(prompt tokens, completion tokens)]. ``floor`` = the chat-template header
    every prompt shares (a slot last used by anyone, including traffic this client never saw,
    honestly shares that much)."""
    best = min(floor, len(tokens))
    for prev, completion in history:
        n = lcp(tokens, prev)
        if n == len(prev):
            n += max(0, int(completion or 0))
        best = max(best, min(n, len(tokens)))
    return best


def pass_rule(depth_rows: list[dict[str, Any]], depths: tuple[int, ...] = SPEC_DEPTHS,
              ratio: float = PASS_RATIO) -> dict[str, Any]:
    """Acceptance check 1. Each row: {depth, n1_tps, n2_total_tps}. PASS only if every spec depth is
    present and measured and 2 runs give >= ratio x the tok/s of 1 run at each."""
    by_depth = {int(r["depth"]): r for r in depth_rows}
    per, missing, failing = [], [], []
    for d in depths:
        r = by_depth.get(d)
        one, two = (r or {}).get("n1_tps"), (r or {}).get("n2_total_tps")
        if not r or not one or two is None:
            missing.append(d)
            per.append({"depth": d, "ratio": None, "pass": False})
            continue
        q = two / one
        ok = q >= ratio
        if not ok:
            failing.append(d)
        per.append({"depth": d, "ratio": round(q, 3), "pass": ok})
    verdict = "INCOMPLETE" if missing else ("PASS" if not failing else "FAIL")
    return {"verdict": verdict, "rule": f"n2_total_tps >= {ratio} x n1_tps at every depth",
            "per_depth": per, "missing": missing, "failing": failing}


def canary_verdict(records: list[dict[str, Any]], *, turns_target: int, min_own_recall: float = 0.8,
                   max_error_frac: float = 0.05) -> dict[str, Any]:
    """Acceptance check 2. FAIL on any bleed (foreign nonce or cache_n over the legit bound). PASS
    needs no bleed AND the planned turns ran AND the model actually repeats its own nonce on recall
    turns (else the nonce detector was never proven able to see anything: WEAK, not PASS)."""
    sent = [r for r in records if not r.get("error")]
    errors = len(records) - len(sent)
    bleed = [r for r in sent if r.get("foreign")]
    over = [r for r in sent if r.get("cache_over_bound")]
    checks = [r for r in sent if r.get("expects_own")]
    own_rate = (sum(1 for r in checks if r.get("own_found")) / len(checks)) if checks else None
    lcp_checked = sum(1 for r in sent if r.get("legit_bound") is not None)
    convo_turns = sum(1 for r in records if r.get("phase") != "fresh")
    reasons = []
    if bleed:
        reasons.append(f"{len(bleed)} response(s) contain another conversation's nonce")
    if over:
        reasons.append(f"{len(over)} request(s) reused more cached tokens than any prompt really shares")
    if bleed or over:
        verdict = "FAIL"
    else:
        if convo_turns < turns_target:
            reasons.append(f"only {convo_turns}/{turns_target} conversation turns ran")
        if records and errors / len(records) > max_error_frac:
            reasons.append(f"{errors}/{len(records)} requests errored")
        if own_rate is None or own_rate < min_own_recall:
            reasons.append(f"own-nonce recall {own_rate} < {min_own_recall}: detector not shown to see nonces")
        if lcp_checked == 0:
            reasons.append("no request had its cache_n checked against the true shared prefix")
        verdict = "WEAK" if reasons else "PASS"
    return {"verdict": verdict, "reasons": reasons, "requests": len(records), "errors": errors,
            "conversation_turns": convo_turns, "turns_target": turns_target,
            "bleed_responses": len(bleed), "cache_over_bound": len(over), "lcp_checked": lcp_checked,
            "own_recall_rate": None if own_rate is None else round(own_rate, 3),
            "tool_calls_emitted": sum(1 for r in sent if r.get("tool_call_emitted")),
            "reasoning_turns": sum(1 for r in sent if r.get("reasoning_chars")),
            "examples": [{k: r.get(k) for k in ("phase", "conv", "turn", "kind", "foreign", "cache_n",
                                                "legit_bound")} for r in (bleed + over)[:10]]}


def preflight_check(pool: dict[str, Any], gpu2_used_mib: int | None, *, actor: str,
                    seat: str = "agent-gpu2", card: str = "gpu2", max_gpu2_mib: int = 4096,
                    bake_port_busy: bool = False) -> tuple[list[str], list[str]]:
    """(info lines, refusal reasons). Empty reasons = safe to start. Read-only."""
    info, reasons = [], []
    info.append(f"pool mode={pool.get('mode')} generated_at={pool.get('generated_at')}")
    for r in pool.get("roles") or []:
        info.append(f"  role {r.get('role'):<11} {r.get('status'):<10} cards={','.join(r.get('cards') or [])} "
                    f"slots={r.get('slots')} ctx/slot={r.get('ctx_per_slot')} profile={r.get('profile_name')}")
    live = [l for l in (pool.get("leases") or []) if l.get("status") in LIVE_LEASE]
    for l in live:
        info.append(f"  live lease {l.get('kind'):<7} {l.get('status'):<9} role={l.get('role')} "
                    f"class={l.get('work_class')} prio={l.get('priority')} holder={l.get('holder')}")
    waiting = [l for l in (pool.get("leases") or []) if l.get("status") in ("queued", "backlogged")]
    info.append(f"  waiting leases: {len(waiting)}")
    info.append(f"  {card} memory used: {gpu2_used_mib} MiB")

    paused = pool.get("actuation_paused")
    if paused:
        by = (paused or {}).get("by")
        if by == actor:
            reasons.append(f"pool actuation is still paused by this bake-off ({actor}) from an earlier run: "
                           "run `stage7_1_bakeoff.sh cleanup` first")
        else:
            reasons.append(f"pool actuation is already paused by {by!r}; this run would resume it at exit "
                           "and undo their stop")
    role = next((r for r in pool.get("roles") or [] if r.get("role") == seat), None)
    if role is None:
        reasons.append(f"role {seat} not in pool state")
    elif role.get("status") not in ("unloaded",) or (role.get("slots") or 0) > 0:
        reasons.append(f"{seat} is {role.get('status')} with {role.get('slots')} slot(s): wait for its idle unload")
    c = next((x for x in pool.get("cards") or [] if x.get("card") == card), None)
    if c is None:
        reasons.append(f"card {card} not in pool state")
    else:
        if c.get("swap_state") != "idle":
            reasons.append(f"{card} swap_state={c.get('swap_state')} (needs idle)")
        act = c.get("actuation") or {}
        if act and not act.get("finished_at"):
            reasons.append(f"{card} has an actuation in flight: {act.get('action')} {act.get('role')} "
                           f"phase={act.get('phase')}")
        if seat in (c.get("swapped_in") or []):
            reasons.append(f"{seat} is swapped in on {card}")
    for l in live:
        if l.get("role") in (seat, "diffusion", "world"):
            reasons.append(f"live {l.get('kind')} on {l.get('role')} (holder {l.get('holder')}): {card} is in use")
        if l.get("holder") == f"operator:{actor}":
            reasons.append(f"an operator hold from an earlier bake-off is still live ({l.get('lease_id')}): "
                           "run cleanup first")
    if gpu2_used_mib is None:
        reasons.append(f"could not read {card} memory (nvidia-smi)")
    elif gpu2_used_mib > max_gpu2_mib:
        reasons.append(f"{card} has {gpu2_used_mib} MiB in use (> {max_gpu2_mib}; only the small world-model "
                       "may stay): something big is still loaded")
    if bake_port_busy:
        reasons.append("the bake-off port already answers: a Bonsai worker is already up")
    return info, reasons


# --------------------------------------------------------------------------- HTTP

class Server:
    def __init__(self, url: str, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def get(self, path: str) -> dict[str, Any]:
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.load(r)

    def tokens_of(self, messages: list[dict], tools: list | None = None) -> list[int] | None:
        """Prompt token ids exactly as the server renders them; None if the build lacks the endpoints."""
        try:
            body: dict[str, Any] = {"messages": messages}
            if tools:
                body["tools"] = tools
            prompt = self.post("/apply-template", body)["prompt"]
            return self.post("/tokenize", {"content": prompt})["tokens"]
        except Exception:  # noqa: BLE001 -- recorded as lcp_unchecked, never fatal
            return None

    def chat(self, body: dict[str, Any]) -> tuple[dict[str, Any] | None, float, str | None]:
        t0 = time.time()
        try:
            return self.post("/v1/chat/completions", body), time.time() - t0, None
        except urllib.error.HTTPError as e:
            return None, time.time() - t0, f"HTTP {e.code}: {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001
            return None, time.time() - t0, repr(e)[:300]


def thinking_kwargs(on: bool, effort: str | None) -> dict[str, Any]:
    if not on:
        return {"enable_thinking": False}
    return {"reasoning_effort": effort} if effort else {"enable_thinking": True}


def filler(rng: random.Random, n_words: int, vocab: list[str] = WORDS) -> str:
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def calibrate_words_per_token(srv: Server, rng: random.Random) -> float:
    sample = filler(rng, 2000)
    toks = srv.post("/tokenize", {"content": sample})["tokens"]
    return 2000 / max(1, len(toks))


# --------------------------------------------------------------------------- bench (check 1)

def _bench_body(messages: list[dict], max_tokens: int) -> dict[str, Any]:
    return {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
            "ignore_eos": True, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}


def _depth_prompt(srv: Server, rng: random.Random, depth: int, wpt: float) -> tuple[list[dict], int]:
    tag = new_nonce(rng)
    head = f"Bench prompt {tag}. Read these field notes, then write as instructed."
    tail = "\n\nNow write a long, detailed story about a lighthouse keeper. Do not stop early."
    words = int((depth - 60) * wpt)
    for _ in range(3):   # measure, correct once or twice: the target is prompt tokens, not words
        msgs = [{"role": "system", "content": head}, {"role": "user", "content": filler(rng, words) + tail}]
        toks = srv.tokens_of(msgs)
        n = len(toks) if toks else depth
        if abs(n - depth) <= max(64, depth * 0.01):
            return msgs, n
        words = max(10, int(words * depth / max(1, n)))
    return msgs, n


def _timed(srv: Server, label: str, msgs: list[dict], max_tokens: int, log: Callable[[str], None]) -> dict:
    t0 = time.time()
    resp, wall, err = srv.chat(_bench_body(msgs, max_tokens))
    t = (resp or {}).get("timings") or {}
    rec = {"label": label, "t_start": t0, "t_end": t0 + wall, "wall_s": round(wall, 2), "error": err,
           "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "prompt_ms": t.get("prompt_ms"),
           "prompt_tps": t.get("prompt_per_second"), "predicted_n": t.get("predicted_n"),
           "predicted_tps": t.get("predicted_per_second")}
    log(f"  {label:<22} prompt_n={rec['prompt_n']} cache_n={rec['cache_n']} prefill_ms={rec['prompt_ms']} "
        f"gen={rec['predicted_n']} tg={rec['predicted_tps']} wall={rec['wall_s']}s err={err}")
    return rec


def _together(fns: list[Callable[[], dict]]) -> list[dict]:
    out: list[dict | None] = [None] * len(fns)
    barrier = threading.Barrier(len(fns))

    def go(i: int) -> None:
        barrier.wait()
        out[i] = fns[i]()
    ths = [threading.Thread(target=go, args=(i,)) for i in range(len(fns))]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return [o or {} for o in out]


def bench_depth(srv: Server, rng: random.Random, depth: int, wpt: float, reps: int, decode_tokens: int,
                log: Callable[[str], None]) -> dict[str, Any]:
    log(f"## depth {depth}")
    a, na = _depth_prompt(srv, rng, depth, wpt)
    b1, nb1 = _depth_prompt(srv, rng, depth, wpt)
    b2, nb2 = _depth_prompt(srv, rng, depth, wpt)
    one_cold = _timed(srv, "n1-cold-prefill", a, 1, log)
    one_warm = [_timed(srv, f"n1-warm-decode-{i}", a, decode_tokens, log) for i in range(reps)]
    two_cold = _together([lambda: _timed(srv, "n2-cold-prefill-A", b1, 1, log),
                          lambda: _timed(srv, "n2-cold-prefill-B", b2, 1, log)])
    two_warm = []
    for i in range(reps):
        pair = _together([lambda i=i: _timed(srv, f"n2-warm-decode-A{i}", b1, decode_tokens, log),
                          lambda i=i: _timed(srv, f"n2-warm-decode-B{i}", b2, decode_tokens, log)])
        two_warm.append(pair)

    def ok(r: dict) -> bool:
        return not r.get("error") and r.get("predicted_tps")
    n1 = [r["predicted_tps"] for r in one_warm if ok(r)]
    n2_sums, n2_wall = [], []
    for pair in two_warm:
        if all(ok(r) for r in pair):
            n2_sums.append(sum(r["predicted_tps"] for r in pair))
            span = max(r["t_end"] for r in pair) - min(r["t_start"] for r in pair)
            n2_wall.append(sum(r["predicted_n"] for r in pair) / span if span > 0 else None)
    # A warm decode only measures "decoding at depth" if its prompt came from the slot's cache.
    warm_hits = [(r.get("cache_n") or 0) >= 0.9 * ((r.get("cache_n") or 0) + (r.get("prompt_n") or 0))
                 for r in one_warm + [x for p in two_warm for x in p] if ok(r)]
    row = {"depth": depth, "prompt_tokens": {"n1": na, "n2": [nb1, nb2]},
           "n1_tps": round(statistics.median(n1), 2) if n1 else None,
           "n2_total_tps": round(statistics.median(n2_sums), 2) if n2_sums else None,
           "n2_wall_aggregate_tps": round(statistics.median([x for x in n2_wall if x]), 2) if any(n2_wall) else None,
           "n2_per_run_tps": [r.get("predicted_tps") for p in two_warm for r in p],
           "n1_prefill_s": round((one_cold.get("prompt_ms") or 0) / 1000, 1) if not one_cold.get("error") else None,
           "n2_prefill_s": [round((r.get("prompt_ms") or 0) / 1000, 1) for r in two_cold],
           "n2_prefill_wall_s": round(max(r.get("t_end", 0) for r in two_cold) - min(r.get("t_start", 0) for r in two_cold), 1),
           "warm_decodes_hit_cache": all(warm_hits) if warm_hits else False,
           "raw": {"n1_cold": one_cold, "n1_warm": one_warm, "n2_cold": two_cold, "n2_warm": two_warm}}
    if row["n1_tps"] and row["n2_total_tps"]:
        row["ratio"] = round(row["n2_total_tps"] / row["n1_tps"], 3)
    log(f"  => n1 {row['n1_tps']} tok/s | n2 total {row['n2_total_tps']} (wall {row['n2_wall_aggregate_tps']}) "
        f"| ratio {row.get('ratio')} | prefill n1 {row['n1_prefill_s']}s n2 {row['n2_prefill_s']}s "
        f"| warm cache hits {row['warm_decodes_hit_cache']}")
    return row


def run_bench(args: argparse.Namespace) -> int:
    srv = Server(args.url)
    rng = random.Random(args.seed)
    log = _logger(args.log)
    props = srv.get("/props")
    slots = props.get("total_slots")
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    log(f"server {args.url} build={props.get('build_info')} slots={slots} ctx/slot={n_ctx} model={props.get('model_path')}")
    depths = [int(x) for x in args.depths.split(",") if x]
    if slots is not None and slots < 2:
        log("REFUSING: the server has fewer than 2 slots, a 2-run measurement is meaningless")
        return 2
    wpt = calibrate_words_per_token(srv, rng)
    rows = []
    for d in depths:
        if n_ctx and d + args.decode_tokens + 64 > n_ctx:
            log(f"## depth {d}: SKIPPED, exceeds ctx/slot {n_ctx}")
            rows.append({"depth": d, "skipped": f"exceeds ctx/slot {n_ctx}"})
            continue
        rows.append(bench_depth(srv, rng, d, wpt, args.reps, args.decode_tokens, log))
        _dump(args.out, {"server": _server_info(props, args.url), "depths": rows, "complete": False})
    verdict = pass_rule([r for r in rows if not r.get("skipped")], tuple(depths))
    _dump(args.out, {"server": _server_info(props, args.url), "depths": rows, "complete": True,
                     "pass_rule": verdict})
    log(f"bench verdict: {verdict['verdict']} {verdict['per_depth']}")
    return 0


# --------------------------------------------------------------------------- canary (check 2)

@dataclass
class Convo:
    cid: str
    nonce: str
    messages: list[dict]
    lock: threading.Lock = field(default_factory=threading.Lock)
    turn: int = 0          # requests sent (a tool follow-up counts)
    step: int = 0          # user turns: picks the turn kind, so tool follow-ups don't starve "think"
    pending_tool: dict | None = None
    full: bool = False


class Canary:
    def __init__(self, srv: Server, rng: random.Random, *, doc_tokens: int, wpt: float, n_ctx: int | None,
                 think_effort: str | None, log: Callable[[str], None], out: str):
        self.srv, self.rng, self.log, self.out = srv, rng, log, out
        self.doc_words = int(doc_tokens * wpt)
        self.n_ctx = n_ctx
        self.think_effort = think_effort
        self.registry: dict[str, str] = {}
        self.floor = 0
        self.history: list[tuple[list[int], int]] = []
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self._n = 0

    def _id(self, prefix: str) -> str:
        with self.lock:
            self._n += 1
            return f"{prefix}{self._n}"

    def _register(self, cid: str) -> str:
        n = new_nonce(self.rng)
        with self.lock:
            self.registry[n] = cid
        return n

    def new_convo(self, prefix: str) -> Convo:
        cid = self._id(prefix)
        nonce = self._register(cid)
        system = (f"{nonce} is the codeword for document {cid}. It belongs to this conversation only.\n\n"
                  f"Document {cid}:\n{filler(self.rng, self.doc_words)}\n\nEnd of document {cid}. "
                  f"Codeword again: {nonce}.")
        return Convo(cid, nonce, [{"role": "system", "content": system}])

    def _user_turn(self, c: Convo) -> tuple[str, str, bool, int, bool]:
        """(kind, user text, thinking on, max_tokens, expects own nonce)"""
        k = c.step % 5
        if k == 0:
            return "recall", "What is this document's codeword? Reply with the codeword only.", False, 32, True
        if k == 1:
            return "chat", "Name three words that occur in the document, then repeat the codeword.", False, 64, True
        if k == 2:
            return ("tool", f"Call the lookup_ledger tool with key '{c.cid}-{c.turn}'. Call the tool, do not answer.",
                    False, 96, False)
        if k == 3:
            return ("think", "Think briefly, then list every codeword or ledger value that appears in this "
                             "conversation, one per line.", True, 384, True)
        return "chat2", "In one sentence, what is the most common word in the document?", False, 64, False

    def _send(self, c: Convo, phase: str, kind: str, messages: list[dict], think: bool, max_tokens: int,
              expects_own: bool) -> dict[str, Any]:
        tokens = self.srv.tokens_of(messages, TOOLS)
        if tokens is not None and self.n_ctx and len(tokens) + max_tokens + 64 > self.n_ctx:
            c.full = True
            return {"skipped": "conversation reached slot ctx"}
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95,
                "cache_prompt": True, "tools": TOOLS, "tool_choice": "auto",
                "chat_template_kwargs": thinking_kwargs(think, self.think_effort)}
        resp, wall, err = self.srv.chat(body)
        msg = ((resp or {}).get("choices") or [{}])[0].get("message") or {}
        timings = (resp or {}).get("timings") or {}
        usage = (resp or {}).get("usage") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        tool_calls = msg.get("tool_calls") or []
        tool_text = json.dumps(tool_calls)
        with self.lock:
            registry = dict(self.registry)
            bound = legit_reuse_bound(tokens, self.history, self.floor) if tokens is not None else None
            if tokens is not None and not err:
                self.history.append((tokens, int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)))
        own = {n for n, owner in registry.items() if owner == c.cid}
        seen = nonces_in(content + "\n" + reasoning + "\n" + tool_text, registry)
        foreign = foreign_nonces(content + "\n" + reasoning + "\n" + tool_text, c.cid, registry)
        cache_n = timings.get("cache_n")
        rec = {"t": round(time.time(), 3), "wall_s": round(wall, 2), "phase": phase, "conv": c.cid,
               "turn": c.turn, "kind": kind, "think": think, "error": err,
               "prompt_tokens": len(tokens) if tokens is not None else None, "cache_n": cache_n,
               "prompt_n": timings.get("prompt_n"), "legit_bound": bound,
               "cache_over_bound": (cache_n is not None and bound is not None and cache_n > bound),
               "cached_tokens_usage": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
               "expects_own": expects_own and not err, "own_found": bool(seen & own),
               "foreign": foreign, "reasoning_chars": len(reasoning), "tool_call_emitted": bool(tool_calls),
               "content": content[:300], "reasoning_tail": reasoning[-300:], "tool_calls": tool_calls[:2]}
        flag = "BLEED" if foreign else ("CACHE>LCP" if rec["cache_over_bound"] else "ok")
        self.log(f"  {phase:<11} {c.cid:<6} t{c.turn:<3} {kind:<7} prompt={rec['prompt_tokens']} "
                 f"cache_n={cache_n} bound={bound} own={rec['own_found']} {flag} err={err} :: {content[:60]!r}")
        with self.lock:
            self.records.append(rec)
        rec["_msg"] = msg
        return rec

    def turn(self, c: Convo, phase: str) -> None:
        with c.lock:
            if c.full:
                return
            if c.pending_tool is not None:          # the follow-up after a tool result
                kind, think, max_tokens, expects = "tool_reply", False, 96, True
                msgs = c.messages + [{"role": "user", "content": "Report the ledger value the tool returned."}]
            else:
                kind, text, think, max_tokens, expects = self._user_turn(c)
                msgs = c.messages + [{"role": "user", "content": text}]
            rec = self._send(c, phase, kind, msgs, think, max_tokens, expects)
            if rec.get("skipped"):
                return
            msg = rec.pop("_msg")
            if rec.get("error"):
                c.turn += 1
                c.step += 1
                c.pending_tool = None   # never leave a tool result unanswered across an error
                return
            c.messages = msgs
            if kind == "tool":
                calls = msg.get("tool_calls") or [{"id": f"call_{c.cid}_{c.turn}", "type": "function",
                                                   "function": {"name": "lookup_ledger",
                                                                "arguments": json.dumps({"key": f"{c.cid}-{c.turn}"})}}]
                calls = [{"id": x.get("id") or f"call_{c.cid}_{c.turn}", "type": "function",
                          "function": {"name": (x.get("function") or {}).get("name") or "lookup_ledger",
                                       "arguments": (x.get("function") or {}).get("arguments") or "{}"}}
                         for x in calls[:1]]
                value = self._register(c.cid)
                c.messages = c.messages + [
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": calls[0]["id"], "content": f"ledger[{c.cid}-{c.turn}] = {value}"}]
                c.pending_tool = {"value": value}
            else:
                if kind == "tool_reply":
                    c.pending_tool = None
                c.messages = c.messages + [{"role": "assistant", "content": msg.get("content") or ""}]
            if kind != "tool_reply":
                c.step += 1
            c.turn += 1

    def measure_floor(self) -> int:
        """Token count of the template header any two of this client's prompts share (two throwaway
        conversations, never sent)."""
        a, b = self.new_convo("H"), self.new_convo("H")
        ta = self.srv.tokens_of(a.messages + [{"role": "user", "content": "x"}], TOOLS)
        tb = self.srv.tokens_of(b.messages + [{"role": "user", "content": "x"}], TOOLS)
        self.floor = lcp(ta, tb) if ta and tb else 0
        return self.floor

    def fresh(self, phase: str = "fresh") -> None:
        c = self.new_convo("F")
        self._send(c, phase, "fresh", c.messages + [{"role": "user", "content":
                   "List every codeword of the form NX... that appears anywhere in this conversation, one per line."}],
                   False, 48, True)

    def flush(self, extra: dict[str, Any]) -> None:
        with self.lock:
            recs = [dict(r) for r in self.records]
        _dump(self.out, {**extra, "records": recs, "nonces": len(self.registry)})


def run_canary(args: argparse.Namespace) -> int:
    srv = Server(args.url, timeout=600)
    rng = random.Random(args.seed)
    log = _logger(args.log)
    props = srv.get("/props")
    slots = props.get("total_slots") or 1
    n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
    log(f"canary on {args.url} build={props.get('build_info')} slots={slots} ctx/slot={n_ctx}")
    can = Canary(srv, rng, doc_tokens=args.doc_tokens, wpt=calibrate_words_per_token(srv, rng), n_ctx=n_ctx,
                 think_effort=args.think_effort, log=log, out=args.out)
    log(f"shared template header (legit reuse floor): {can.measure_floor()} tokens")
    deadline = time.time() + args.deadline_min * 60
    meta = {"server": _server_info(props, args.url), "doc_tokens": args.doc_tokens,
            "plan": {"pairs_turns_per_conv": args.pair_turns, "interleave_turns_per_conv": args.interleave_turns,
                     "fresh_every": args.fresh_every}}
    stopped_early = None

    log(f"## phase pairs: 2 conversations, {args.pair_turns} turns each, both sent at the same moment")
    pair = [can.new_convo("P"), can.new_convo("P")]
    for r in range(args.pair_turns):
        if time.time() > deadline:
            stopped_early = f"deadline in pairs at round {r}"
            break
        _together([lambda c=c: can.turn(c, "pairs") for c in pair])
        if args.fresh_every and r % args.fresh_every == args.fresh_every - 1:
            can.fresh()
        if args.pace:
            time.sleep(args.pace)
        can.flush(meta)

    if not stopped_early:
        workers = 2   # 4 conversations over 2 concurrent requests: the --parallel 2 layout under test
        log(f"## phase interleave: 4 conversations over {workers} concurrent workers ({slots} slots), "
            f"{args.interleave_turns} turns each; consecutive requests switch conversations")
        quad = [can.new_convo("Q") for _ in range(4)]
        order = [c for _ in range(args.interleave_turns) for c in quad]
        idx = {"i": 0}
        qlock = threading.Lock()

        def worker() -> None:
            nonlocal stopped_early
            while True:
                with qlock:
                    if idx["i"] >= len(order) or time.time() > deadline:
                        if idx["i"] < len(order):
                            stopped_early = f"deadline in interleave at request {idx['i']}/{len(order)}"
                        return
                    c = order[idx["i"]]
                    idx["i"] += 1
                    k = idx["i"]
                can.turn(c, "interleave")
                if args.fresh_every and k % (args.fresh_every * 4) == 0:
                    can.fresh()
                if args.pace:
                    time.sleep(args.pace)
                if k % 8 == 0:
                    can.flush(meta)
        ths = [threading.Thread(target=worker) for _ in range(workers)]
        [t.start() for t in ths]
        [t.join() for t in ths]

    target = 2 * args.pair_turns + 4 * args.interleave_turns
    verdict = canary_verdict(can.records, turns_target=target)
    if stopped_early:
        verdict["stopped_early"] = stopped_early
    can.flush({**meta, "verdict": verdict, "complete": True})
    log(f"canary verdict: {verdict['verdict']} {verdict['reasons']} requests={verdict['requests']} "
        f"bleed={verdict['bleed_responses']} cache_over_bound={verdict['cache_over_bound']} "
        f"own_recall={verdict['own_recall_rate']} tool_calls={verdict['tool_calls_emitted']}")
    return 0


# --------------------------------------------------------------------------- summary + field note

def summarize(results_dir: Path) -> dict[str, Any]:
    passes = {}
    for bench in sorted(results_dir.glob("*/bench.json")):
        label = bench.parent.name
        passes.setdefault(label, {})["bench"] = json.loads(bench.read_text())
    for canary in sorted(results_dir.glob("*/canary.json")):
        passes.setdefault(canary.parent.name, {})["canary"] = json.loads(canary.read_text())
    out: dict[str, Any] = {"results_dir": str(results_dir), "passes": {}}
    for label, p in passes.items():
        b, c = p.get("bench") or {}, p.get("canary") or {}
        rows = [r for r in b.get("depths") or [] if not r.get("skipped")]
        depths = tuple(r["depth"] for r in b.get("depths") or [])
        out["passes"][label] = {
            "server": b.get("server") or c.get("server"),
            "bench": pass_rule(rows, depths) if b else {"verdict": "NOT_RUN"},
            "bench_rows": [{k: r.get(k) for k in ("depth", "n1_tps", "n2_total_tps", "n2_wall_aggregate_tps",
                                                  "ratio", "n1_prefill_s", "n2_prefill_s", "warm_decodes_hit_cache",
                                                  "skipped")} for r in b.get("depths") or []],
            "canary": (c.get("verdict") or canary_verdict(c.get("records") or [], turns_target=1)) if c
            else {"verdict": "NOT_RUN"},
        }
    return out


def field_note_table(summary: dict[str, Any]) -> str:
    lines = []
    for label, p in summary["passes"].items():
        s = p.get("server") or {}
        lines.append(f"### {label}: {s.get('model_path')} build {s.get('build_info')}, "
                     f"{s.get('slots')} slots x {s.get('ctx_per_slot')}")
        lines.append("")
        lines.append("| Depth | 1 run tok/s | 2 runs total tok/s (wall) | Ratio | Pass (>= 1.3) | Prefill 1 run (s) | Prefill 2 runs (s) |")
        lines.append("|---:|---:|---:|---:|:--:|---:|---:|")
        per = {x["depth"]: x for x in p["bench"].get("per_depth") or []}
        for r in p["bench_rows"]:
            if r.get("skipped"):
                lines.append(f"| {r['depth']:,} | skipped: {r['skipped']} | | | | | |")
                continue
            ok = per.get(r["depth"], {}).get("pass")
            lines.append(f"| {r['depth']:,} | {r.get('n1_tps')} | {r.get('n2_total_tps')} ({r.get('n2_wall_aggregate_tps')}) "
                         f"| {r.get('ratio')} | {'yes' if ok else 'NO'} | {r.get('n1_prefill_s')} | {r.get('n2_prefill_s')} |")
        lines.append("")
        lines.append(f"Bench verdict: **{p['bench'].get('verdict')}**")
        c = p["canary"]
        lines.append(f"Canary verdict: **{c.get('verdict')}** -- requests {c.get('requests')}, conversation turns "
                     f"{c.get('conversation_turns')}/{c.get('turns_target')}, bleed {c.get('bleed_responses')}, "
                     f"cache_n over true prefix {c.get('cache_over_bound')}, own-nonce recall {c.get('own_recall_rate')}, "
                     f"tool calls emitted {c.get('tool_calls_emitted')}, reasoning turns {c.get('reasoning_turns')}"
                     + (f"; reasons: {c.get('reasons')}" if c.get("reasons") else ""))
        lines.append("")
    return "\n".join(lines)


def run_summarize(args: argparse.Namespace) -> int:
    d = Path(args.results_dir)
    s = summarize(d)
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    (d / "fieldnote_draft.md").write_text("# Stage 7.1 bake-off: draft table\n\n" + field_note_table(s) + "\n")
    print(field_note_table(s))
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    pool = json.loads(Path(args.pool_json).read_text())
    used = None if args.gpu2_used_mib in (None, "", "unknown") else int(float(args.gpu2_used_mib))
    info, reasons = preflight_check(pool, used, actor=args.actor, max_gpu2_mib=args.max_gpu2_mib,
                                    bake_port_busy=args.bake_port_busy)
    print("\n".join(info))
    if reasons:
        print("REFUSE:")
        print("\n".join(f"  - {r}" for r in reasons))
        return 3
    print("preflight: OK")
    return 0


# --------------------------------------------------------------------------- plumbing

def _server_info(props: dict[str, Any], url: str) -> dict[str, Any]:
    return {"url": url, "build_info": props.get("build_info"), "model_path": props.get("model_path"),
            "slots": props.get("total_slots"),
            "ctx_per_slot": (props.get("default_generation_settings") or {}).get("n_ctx")}


def _dump(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


def _logger(path: str | None) -> Callable[[str], None]:
    lock = threading.Lock()

    def log(s: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {s}"
        with lock:
            print(line, flush=True)
            if path:
                with open(path, "a") as f:
                    f.write(line + "\n")
    return log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight-check")
    p.add_argument("--pool-json", required=True)
    p.add_argument("--gpu2-used-mib")
    p.add_argument("--max-gpu2-mib", type=int, default=4096)
    p.add_argument("--actor", default="stage7-1-bakeoff")
    p.add_argument("--bake-port-busy", action="store_true")
    p.set_defaults(fn=run_preflight)
    p = sub.add_parser("bench")
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log")
    p.add_argument("--depths", default=",".join(str(d) for d in SPEC_DEPTHS))
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--seed", type=int)
    p.set_defaults(fn=run_bench)
    p = sub.add_parser("canary")
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log")
    p.add_argument("--doc-tokens", type=int, default=4600)
    p.add_argument("--pair-turns", type=int, default=100, help="turns per conversation, 2 conversations")
    p.add_argument("--interleave-turns", type=int, default=50, help="turns per conversation, 4 conversations")
    p.add_argument("--fresh-every", type=int, default=10)
    p.add_argument("--think-effort", default="medium",
                   help="reasoning_effort for thinking turns ('' = enable_thinking true)")
    p.add_argument("--deadline-min", type=float, default=50)
    p.add_argument("--pace", type=float, default=0.0, help="sleep between rounds (live lanes)")
    p.add_argument("--seed", type=int)
    p.set_defaults(fn=run_canary)
    p = sub.add_parser("summarize")
    p.add_argument("--results-dir", required=True)
    p.set_defaults(fn=run_summarize)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
