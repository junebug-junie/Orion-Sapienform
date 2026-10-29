#!/usr/bin/env python3
"""Bench one GPU's llama-server at several power limits: speed, joules per token, temperature.

Question it answers (body economy, 2026-10-29): does capping a card's watts buy energy per token
without costing much speed? If joules per generated token barely moves, power caps are not worth
building. Run it on the card's own host, as root (setting a power limit needs root):

    sudo python3 scripts/gpu_power_bench.py --gpu 2 --port 8016

Stdlib only, so it runs with the host's python3 and no Orion venv.

What it does, per cap (default 250, 200, 150, 100 W):
  1. sets the cap (nvidia-smi -pl), waits for the card to settle;
  2. sends the same fixed prompt REPEATS times to one slot, prompt cache off, a fixed token count;
  3. integrates the card's power over the reading phase and the writing phase of each request.

It takes no GPU pool hold (a pool hold that lands on agent-gpu2 also pins gpu1). Instead a request
goes out only when every slot is idle and the card is near idle power, and a sample is thrown away
if any other slot works during it, if the server's decode counter moved by more than our own
tokens, or if the request waited before it started. Other tenants of the card that are not this
server (world-model, diffusion) are NOT detectable from the server; they show as power noise.

The card's starting limit is restored on every exit: normal end, error, Ctrl+C, SIGTERM, SIGHUP,
SIGQUIT. Signals are ignored while the restore runs. Only SIGKILL skips it; the restore command
is printed at start for that case. It refuses to start while another run holds the card's lock,
or while the card is not at its default limit (someone else may be capping it).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import http.client
import json
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPT_SENTENCES = 60  # ~2k tokens (220 measured 7.4k on Bonsai 2); the same bytes on every run
SMI_TIMEOUT_S = 30
# Request round trip beyond the server's own prompt_ms + predicted_ms: tokenizing, HTTP, JSON.
# An idle card measured 0.84 s of it on circe gpu2 (2026-10-29, 2k-token prompt, 512 tokens out);
# a request queued behind real agent work waits seconds, so more than this means its windows are wrong.
MAX_OVERHEAD_S = 2.0
SAMPLE_STALE_S = 5.0  # no fresh power reading for this long = the sampler is dead; abort
HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


class BenchAbort(RuntimeError):
    """The bench cannot produce trustworthy numbers any more; stop and restore the card."""


# What a server that unloads, stalls or dies mid-request raises through urllib on Python 3.12:
# connect errors arrive as URLError, but a read timeout is a bare TimeoutError (OSError), a
# half-sent reply is an HTTPException or bad JSON, and a reply without timings is a KeyError.
SERVER_GONE = (urllib.error.URLError, OSError, http.client.HTTPException, json.JSONDecodeError, KeyError)


def fixed_prompt() -> str:
    lines = [
        f"Note {i}: the walkway light turned on at {i % 24:02d}:{(i * 7) % 60:02d}, "
        f"the cabinet read {30 + (i % 9)} degrees, and nothing else changed."
        for i in range(PROMPT_SENTENCES)
    ]
    return "\n".join(lines) + "\nWrite a long, plain summary of these notes:\n"


def energy_joules(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    """Integral of watts over [t0, t1] by the trapezoid rule, from (time, watts) samples.

    Samples are linearly interpolated at the window edges. None when the window is not covered
    by samples on both sides (the sampler fell behind or stopped): no number beats a made-up one.
    """
    if t1 <= t0 or len(samples) < 2 or samples[0][0] > t0 or samples[-1][0] < t1:
        return None

    def watts_at(t: float) -> float:
        for (ta, wa), (tb, wb) in zip(samples, samples[1:]):
            if ta <= t <= tb:
                return wa if tb == ta else wa + (wb - wa) * (t - ta) / (tb - ta)
        raise ValueError(t)

    points = [(t0, watts_at(t0))] + [s for s in samples if t0 < s[0] < t1] + [(t1, watts_at(t1))]
    return sum((tb - ta) * (wa + wb) / 2 for (ta, wa), (tb, wb) in zip(points, points[1:]))


@dataclass
class Sample:
    cap_w: int
    rep: int
    prompt_n: int
    prompt_s: float
    gen_n: int
    gen_s: float
    prompt_tok_s: float
    gen_tok_s: float
    prompt_j_per_tok: float | None
    gen_j_per_tok: float | None
    gen_avg_w: float | None
    max_temp_c: int
    max_sm_mhz: int
    waited_s: float = 0.0


class Sampler(threading.Thread):
    """Reads power, temperature and SM clock from nvidia-smi as fast as it answers (~15 Hz).

    Each reading is stamped at the midpoint of its nvidia-smi call, not at its end, so the trace is
    not shifted late by the call's own cost. Errors are counted, never fatal to the thread; the main
    loop aborts when readings stop arriving (`fresh`)."""

    def __init__(self, gpu: int) -> None:
        super().__init__(daemon=True)
        self.gpu = gpu
        self.power: list[tuple[float, float]] = []
        self.temp: list[tuple[float, int]] = []
        self.clock: list[tuple[float, int]] = []
        self.errors = 0
        self._halt = threading.Event()

    def run(self) -> None:
        cmd = ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=power.draw,temperature.gpu,clocks.sm",
               "--format=csv,noheader,nounits"]
        while not self._halt.is_set():
            try:
                t0 = time.monotonic()
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                     start_new_session=True).stdout.strip()
                t = (t0 + time.monotonic()) / 2
                w, c, mhz = (x.strip() for x in out.split(","))
                self.power.append((t, float(w)))
                self.temp.append((t, int(c)))
                self.clock.append((t, int(mhz)))
            except Exception:  # noqa: BLE001 -- keep sampling; staleness is checked by the main loop
                self.errors += 1
                self._halt.wait(0.5)

    def stop(self) -> None:
        self._halt.set()

    def fresh(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.is_alive() and bool(self.power) and now - self.power[-1][0] < SAMPLE_STALE_S

    def window(self, series: list, t0: float, t1: float) -> list:
        return [v for t, v in series if t0 <= t <= t1]


def smi(*args: str) -> str:
    # Own session: a Ctrl+C at the terminal must not kill a half-done `-pl`.
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True,
                          timeout=SMI_TIMEOUT_S, start_new_session=True).stdout


def read_limits(gpu: int) -> tuple[float, float, float, float]:
    """(current, default, min, max) watts."""
    out = smi("-i", str(gpu),
              "--query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit",
              "--format=csv,noheader,nounits")
    cur, default, lo, hi = (float(x) for x in out.strip().split(","))
    return cur, default, lo, hi


def set_limit(gpu: int, watts: float) -> None:
    smi("-i", str(gpu), "-pl", f"{watts:.0f}")


def restore_limit(gpu: int, watts: float, attempts: int = 3) -> bool:
    """Put the card back. Signals are ignored for the duration, and every failure (BaseException
    included) is retried, so a second Ctrl+C or a dropped terminal cannot leave the card capped."""
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in HANDLED_SIGNALS}
    try:
        for i in range(attempts):
            try:
                set_limit(gpu, watts)
                return True
            except BaseException as exc:  # noqa: BLE001
                print(f"restore attempt {i + 1} failed: {exc!r}", file=sys.stderr)
        return False
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def http_json(url: str, body: dict | None = None, timeout: float = 600) -> object:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def wait_for_server(base: str, wait_s: float) -> bool:
    """True once the server answers /health. gpu2's agent model is unloaded whenever image
    generation (diffusion) takes the card, and reloads after it has been quiet for a while."""
    deadline = time.monotonic() + wait_s
    told = False
    while True:
        try:
            http_json(f"{base}/health", timeout=5)
            return True
        except Exception:  # noqa: BLE001
            if time.monotonic() >= deadline:
                return False
            if not told:
                print(f"model server at {base} is not answering (on gpu2 this usually means image "
                      f"generation has the card); waiting up to {wait_s / 60:.0f} min for it", flush=True)
                told = True
            time.sleep(10)


def busy_slots(base: str, ignore: int | None = None) -> list[int]:
    """Ids of slots doing work. Raises BenchAbort if this server's /slots cannot say."""
    slots = http_json(f"{base}/slots", timeout=10)
    if not isinstance(slots, list) or not slots or any("is_processing" not in s for s in slots):
        raise BenchAbort(f"{base}/slots does not report is_processing per slot; cannot detect sharing")
    return [s["id"] for s in slots if s["is_processing"] and s["id"] != ignore]


_DECODE_RE = re.compile(r"^llamacpp:(tokens_predicted_total|prompt_tokens_total)\s+([0-9.e+]+)$", re.M)


def token_counter(base: str) -> float | None:
    """Server-wide prompt + predicted token total from /metrics; None when metrics are off."""
    try:
        req = urllib.request.Request(f"{base}/metrics")
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode()
    except Exception:  # noqa: BLE001 -- metrics are an extra check, not required
        return None
    found = _DECODE_RE.findall(text)
    return sum(float(v) for _, v in found) if len(found) == 2 else None


class SlotWatch(threading.Thread):
    """Polls /slots during one request; `.dirty` is True if any other slot did work meanwhile.
    Three failed polls in a row set `.broken` (abort the run, do not keep dropping samples)."""

    def __init__(self, base: str, our_slot: int) -> None:
        super().__init__(daemon=True)
        self.base, self.our_slot = base, our_slot
        self.dirty = False
        self.broken = False
        self._halt = threading.Event()

    def _poll(self, fails: int) -> int:
        try:
            self.dirty |= bool(busy_slots(self.base, ignore=self.our_slot))
            return 0
        except Exception:  # noqa: BLE001
            self.dirty = True
            self.broken |= fails + 1 >= 3
            return fails + 1

    def run(self) -> None:
        fails = self._poll(0)
        while not self._halt.wait(0.2):
            fails = self._poll(fails)
        self._poll(fails)  # one last look after the request returned

    def stop(self) -> None:
        self._halt.set()


def wait_until_quiet(base: str, sampler: Sampler, idle_w: float, timeout_s: float) -> bool:
    """True once every slot is idle and the card drew under idle_w across the last 3 s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not sampler.fresh():
            raise BenchAbort(f"power sampler stopped ({sampler.errors} errors)")
        now = time.monotonic()
        recent = [(t, w) for t, w in sampler.power if t >= now - 3]
        covered = len(recent) >= 2 and recent[-1][0] - recent[0][0] >= 2.0
        if covered and max(w for _, w in recent) < idle_w and not busy_slots(base):
            return True
        time.sleep(0.5)
    return False


def one_request(base: str, prompt: str, n_predict: int, slot: int) -> tuple[float, float, dict]:
    t_send = time.monotonic()
    res = http_json(f"{base}/completion", {
        "prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "temperature": 0,
        "cache_prompt": False, "id_slot": slot,
    })
    return t_send, time.monotonic(), res["timings"]


def waited_s(t_send: float, t_recv: float, timings: dict) -> float:
    """Round-trip time not spent in the server's own prompt and generation clocks."""
    return (t_recv - t_send) - (timings["prompt_ms"] + timings["predicted_ms"]) / 1000


def to_sample(cap: int, rep: int, sampler: Sampler, t_recv: float, timings: dict) -> Sample:
    """Windows are anchored to the reply's arrival: writing ends just before it, reading before that.
    The small response overhead after writing is ignored (it sits on the idle side of the window)."""
    prompt_s = timings["prompt_ms"] / 1000
    gen_s = timings["predicted_ms"] / 1000
    t_end = t_recv
    t_mid = t_end - gen_s
    t_start = t_mid - prompt_s
    e_prompt = energy_joules(sampler.power, t_start, t_mid)
    e_gen = energy_joules(sampler.power, t_mid, t_end)
    pn, gn = int(timings["prompt_n"]), int(timings["predicted_n"])
    return Sample(
        cap_w=cap, rep=rep, prompt_n=pn, prompt_s=round(prompt_s, 3), gen_n=gn, gen_s=round(gen_s, 3),
        prompt_tok_s=round(pn / prompt_s, 1) if prompt_s else 0.0,
        gen_tok_s=round(gn / gen_s, 2) if gen_s else 0.0,
        prompt_j_per_tok=round(e_prompt / pn, 4) if e_prompt is not None and pn else None,
        gen_j_per_tok=round(e_gen / gn, 3) if e_gen is not None and gn else None,
        gen_avg_w=round(e_gen / gen_s, 1) if e_gen is not None and gen_s else None,
        max_temp_c=max(sampler.window(sampler.temp, t_start, t_end), default=0),
        max_sm_mhz=max(sampler.window(sampler.clock, t_start, t_end), default=0),
    )


def summarize(samples: list[Sample], caps: list[int]) -> list[dict]:
    """One row per requested cap. `*_vs_top` compares to caps[0] (the highest, normally the card's
    default) and is None when that cap has no clean samples: no silent change of baseline."""
    rows = []
    for cap in caps:
        got = [s for s in samples if s.cap_w == cap]

        def med(field: str) -> float | None:
            vals = [getattr(s, field) for s in got if getattr(s, field) is not None]
            return round(statistics.median(vals), 3) if vals else None

        rows.append({"cap_w": cap, "n": len(got), "prompt_tok_s": med("prompt_tok_s"),
                     "gen_tok_s": med("gen_tok_s"), "gen_avg_w": med("gen_avg_w"),
                     "prompt_j_per_tok": med("prompt_j_per_tok"), "gen_j_per_tok": med("gen_j_per_tok"),
                     "max_temp_c": max((s.max_temp_c for s in got), default=None),
                     "max_sm_mhz": max((s.max_sm_mhz for s in got), default=None)})
    top = rows[0] if rows else None
    for r in rows:
        for key in ("gen_tok_s", "prompt_tok_s", "gen_j_per_tok"):
            r[f"{key}_vs_top"] = (round(r[key] / top[key], 3)
                                  if top and r[key] is not None and top[key] else None)
    return rows


def write_outputs(out: Path, samples: list[Sample], caps: list[int], meta: dict) -> list[dict]:
    with (out / "samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(Sample.__dataclass_fields__))
        w.writeheader()
        w.writerows(asdict(s) for s in samples)
    rows = summarize(samples, caps)
    (out / "summary.json").write_text(json.dumps({"meta": meta, "by_cap": rows}, indent=2))
    return rows


def run_caps(args: argparse.Namespace, base: str, caps: list[int], sampler: Sampler,
             samples: list[Sample], dropped: list[dict], save) -> None:
    prompt = fixed_prompt()
    for cap in caps:
        set_limit(args.gpu, cap)
        print(f"\n== {cap} W ==", flush=True)
        time.sleep(args.settle_s)
        one_request(base, prompt, 16, args.slot)  # warm-up at this cap, not recorded
        rep = attempts = 0
        while rep < args.repeats and attempts < args.repeats + args.max_retries:
            attempts += 1
            if not wait_until_quiet(base, sampler, args.idle_w, timeout_s=args.quiet_timeout_s):
                dropped.append({"cap_w": cap, "why": "never_quiet"})
                print(f"  card never went quiet in {args.quiet_timeout_s:.0f} s; skipping one attempt",
                      flush=True)
                continue
            before = token_counter(base)
            watch = SlotWatch(base, args.slot)
            watch.start()
            try:
                t_send, t_recv, timings = one_request(base, prompt, args.n_predict, args.slot)
            finally:
                watch.stop()
                watch.join()
            if watch.broken:
                raise BenchAbort("/slots stopped answering during a request")
            after = token_counter(base)
            ours = timings["prompt_n"] + timings["predicted_n"]
            why = None
            if watch.dirty:
                why = "other_slot_busy"
            elif before is not None and after is not None and after - before > ours + 16:
                why = f"server_counted_{after - before - ours:.0f}_extra_tokens"
            elif waited_s(t_send, t_recv, timings) > MAX_OVERHEAD_S:
                why = f"request_waited_{waited_s(t_send, t_recv, timings):.2f}s"
            if why:
                dropped.append({"cap_w": cap, "why": why})
                print(f"  sample thrown away: {why}", flush=True)
                continue
            time.sleep(0.5)  # let the sampler cover the end of the window
            s = to_sample(cap, rep, sampler, t_recv, timings)
            s.waited_s = round(waited_s(t_send, t_recv, timings), 3)
            samples.append(s)
            save()
            rep += 1
            print(f"  rep {rep}: read {s.prompt_tok_s} tok/s, write {s.gen_tok_s} tok/s, "
                  f"{s.gen_avg_w} W, {s.gen_j_per_tok} J/token, {s.max_temp_c} C", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, required=True, help="nvidia-smi index of the card")
    ap.add_argument("--port", type=int, required=True, help="llama-server port on this host")
    ap.add_argument("--caps", default="250,200,150,100", help="watts, comma separated, highest first")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=512)  # ~11 s of writing at 46 tok/s
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--idle-w", type=float, default=80.0, help="card must draw under this before a request")
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--quiet-timeout-s", type=float, default=180.0)
    ap.add_argument("--max-retries", type=int, default=6, help="per cap, for samples thrown away as dirty")
    ap.add_argument("--server-wait-s", type=float, default=3600.0,
                    help="how long to wait for the model server to come up before starting")
    ap.add_argument("--lock-dir", default="/run")
    ap.add_argument("--out", default=f"/tmp/gpu-power-bench-{time.strftime('%Y%m%dT%H%M%S')}")
    args = ap.parse_args(argv)

    base = f"http://127.0.0.1:{args.port}"
    caps = [int(c) for c in args.caps.split(",")]
    original, default, lo, hi = read_limits(args.gpu)
    bad = [c for c in caps if not lo <= c <= hi]
    if bad:
        print(f"caps {bad} outside this card's range {lo:.0f}-{hi:.0f} W", file=sys.stderr)
        return 2
    if original != default:
        print(f"card {args.gpu} is at {original:.0f} W, not its default {default:.0f} W: something else "
              f"may be capping it. Not starting.", file=sys.stderr)
        return 2
    lock = open(Path(args.lock_dir) / f"gpu-power-bench-{args.gpu}.lock", "w")  # noqa: SIM115
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another bench already holds card {args.gpu}", file=sys.stderr)
        return 2
    try:
        up = wait_for_server(base, args.server_wait_s)
    except KeyboardInterrupt:
        print(f"\ninterrupted while waiting; nothing was changed on card {args.gpu}", file=sys.stderr)
        return 130
    if not up:
        print(f"model server at {base} never came up; nothing was changed on card {args.gpu}", file=sys.stderr)
        return 3
    try:
        busy_slots(base)  # refuses a server whose /slots cannot show sharing, before any cap is set
    except BenchAbort as exc:
        print(f"{exc}; nothing was changed on card {args.gpu}", file=sys.stderr)
        return 2
    except SERVER_GONE as exc:
        print(f"model server went away before the start ({exc!r}); nothing was changed", file=sys.stderr)
        return 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"card {args.gpu}: limit {original:.0f} W (range {lo:.0f}-{hi:.0f}); results in {out}")
    print(f"if this process is SIGKILLed, restore with: sudo nvidia-smi -i {args.gpu} -pl {original:.0f}")

    def _raise(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in HANDLED_SIGNALS[1:]:  # SIGINT already raises KeyboardInterrupt
        signal.signal(sig, _raise)

    meta = {"gpu": args.gpu, "port": args.port, "original_limit_w": original, "caps": caps,
            "n_predict": args.n_predict, "prompt_chars": len(fixed_prompt()), "dropped": []}
    try:
        meta["model"] = http_json(f"{base}/props", timeout=10).get("model_path")
    except Exception:  # noqa: BLE001
        meta["model"] = None
    samples: list[Sample] = []
    sampler = Sampler(args.gpu)
    sampler.start()
    try:
        try:
            run_caps(args, base, caps, sampler, samples, meta["dropped"],
                     save=lambda: write_outputs(out, samples, caps, meta))
        except SERVER_GONE as exc:
            # Image generation took the card mid-run (the server unloads or stalls), or it died.
            meta["aborted"] = f"model server went away: {exc!r}"
            print(f"\nstopped early, {meta['aborted']}; keeping the {len(samples)} samples so far",
                  file=sys.stderr)
        except BenchAbort as exc:
            meta["aborted"] = str(exc)
            print(f"\nstopped early: {exc}", file=sys.stderr)
    finally:
        if restore_limit(args.gpu, original):
            print(f"\nrestored card {args.gpu} to {original:.0f} W")
        else:
            print(f"\nRESTORE FAILED; run: sudo nvidia-smi -i {args.gpu} -pl {original:.0f}", file=sys.stderr)
        sampler.stop()
        write_outputs(out, samples, caps, meta)

    rows = summarize(samples, caps)
    print("\ncap  read tok/s  write tok/s  write W  J/token(write)  vs top cap (speed / energy)  max C")
    for r in rows:
        print(f"{r['cap_w']:>4} {r['prompt_tok_s']!s:>10} {r['gen_tok_s']!s:>12} {r['gen_avg_w']!s:>8} "
              f"{r['gen_j_per_tok']!s:>15}  {r['gen_tok_s_vs_top']!s:>8} / {r['gen_j_per_tok_vs_top']!s:<8}"
              f"          {r['max_temp_c']}")
    print(f"\nwritten: {out / 'samples.csv'}, {out / 'summary.json'}; dropped samples: {len(meta['dropped'])}")
    return 1 if meta.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
