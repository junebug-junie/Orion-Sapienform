#!/usr/bin/env python3
"""Bench one GPU's llama-server at several power limits: speed, joules per token, temperature.

Question it answers (body economy, 2026-10-29): does capping a card's watts buy energy per token
without costing much speed? If joules per generated token barely moves, power caps are not worth
building. Run it on the card's own host, as root (setting a power limit needs root):

    sudo python3 scripts/gpu_power_bench.py --gpu 2 --port 8016

Stdlib only, so it runs with the host's python3 and no Orion venv.

What it does, per cap (default 250, 200, 150, 100 W):
  1. sets the cap (nvidia-smi -pl), waits for the card to settle;
  2. sends the same fixed prompt REPEATS times to slot 0, prompt cache off, a fixed number of tokens;
  3. integrates the card's power over the reading phase and the writing phase of each request.

It takes no GPU pool hold (a pool hold that lands on agent-gpu2 also pins gpu1). Instead every
request is pinned to slot 0 and slot 1 is watched: if the server starts other work, or the card
draws more than the idle threshold before a request, that sample is thrown away and repeated.
Other tenants of the same card that are not this server (world-model, diffusion) are NOT detectable
from the server; their draw would show up as noise in the power samples.

The card's original limit is restored on every exit: normal end, error, Ctrl+C, SIGTERM, SIGHUP.
Only SIGKILL skips it; the restore command is printed at start for that case.
"""
from __future__ import annotations

import argparse
import csv
import json
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

PROMPT_SENTENCES = 60  # ~2k tokens (220 measured 7.4k on Bonsai 2); the same bytes on every run


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


class Sampler(threading.Thread):
    """Reads power, temperature and SM clock from nvidia-smi as fast as it answers (~10-20 Hz)."""

    def __init__(self, gpu: int) -> None:
        super().__init__(daemon=True)
        self.gpu = gpu
        self.power: list[tuple[float, float]] = []
        self.temp: list[tuple[float, int]] = []
        self.clock: list[tuple[float, int]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        cmd = ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=power.draw,temperature.gpu,clocks.sm",
               "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            t = time.monotonic()
            try:
                w, c, mhz = (x.strip() for x in out.split(","))
                self.power.append((t, float(w)))
                self.temp.append((t, int(c)))
                self.clock.append((t, int(mhz)))
            except ValueError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def window(self, series: list, t0: float, t1: float) -> list:
        return [v for t, v in series if t0 <= t <= t1]


def smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


def read_limits(gpu: int) -> tuple[float, float, float]:
    out = smi("-i", str(gpu), "--query-gpu=power.limit,power.min_limit,power.max_limit",
              "--format=csv,noheader,nounits")
    cur, lo, hi = (float(x) for x in out.strip().split(","))
    return cur, lo, hi


def set_limit(gpu: int, watts: float) -> None:
    smi("-i", str(gpu), "-pl", f"{watts:.0f}")


def http_json(url: str, body: dict | None = None, timeout: float = 600) -> object:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def other_slots_busy(base: str, our_slot: int) -> bool:
    slots = http_json(f"{base}/slots", timeout=10)
    return any(s.get("is_processing") and s.get("id") != our_slot for s in slots)


class SlotWatch(threading.Thread):
    """Polls /slots during one request; `.dirty` is True if any other slot did work meanwhile."""

    def __init__(self, base: str, our_slot: int) -> None:
        super().__init__(daemon=True)
        self.base, self.our_slot = base, our_slot
        self.dirty = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self.dirty |= other_slots_busy(self.base, self.our_slot)
            except Exception:  # noqa: BLE001 -- an unreadable /slots cannot vouch for a clean sample
                self.dirty = True

    def stop(self) -> None:
        self._stop.set()


def wait_until_quiet(base: str, sampler: Sampler, our_slot: int, idle_w: float, timeout_s: float) -> bool:
    """True once the other slots are idle and the card drew under idle_w for the last 3 s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        now = time.monotonic()
        recent = sampler.window(sampler.power, now - 3, now)
        if len(recent) >= 5 and max(recent) < idle_w and not other_slots_busy(base, our_slot):
            return True
        time.sleep(1)
    return False


def one_request(base: str, prompt: str, n_predict: int, slot: int) -> tuple[float, float, dict]:
    t_send = time.monotonic()
    res = http_json(f"{base}/completion", {
        "prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "temperature": 0,
        "cache_prompt": False, "id_slot": slot,
    })
    return t_send, time.monotonic(), res["timings"]


def to_sample(cap: int, rep: int, sampler: Sampler, t_send: float, timings: dict) -> Sample:
    prompt_s = timings["prompt_ms"] / 1000
    gen_s = timings["predicted_ms"] / 1000
    t_mid = t_send + prompt_s
    t_end = t_mid + gen_s
    e_prompt = energy_joules(sampler.power, t_send, t_mid)
    e_gen = energy_joules(sampler.power, t_mid, t_end)
    pn, gn = int(timings["prompt_n"]), int(timings["predicted_n"])
    return Sample(
        cap_w=cap, rep=rep, prompt_n=pn, prompt_s=round(prompt_s, 3), gen_n=gn, gen_s=round(gen_s, 3),
        prompt_tok_s=round(pn / prompt_s, 1) if prompt_s else 0.0,
        gen_tok_s=round(gn / gen_s, 2) if gen_s else 0.0,
        prompt_j_per_tok=round(e_prompt / pn, 4) if e_prompt is not None and pn else None,
        gen_j_per_tok=round(e_gen / gn, 3) if e_gen is not None and gn else None,
        gen_avg_w=round(e_gen / gen_s, 1) if e_gen is not None and gen_s else None,
        max_temp_c=max(sampler.window(sampler.temp, t_send, t_end), default=0),
        max_sm_mhz=max(sampler.window(sampler.clock, t_send, t_end), default=0),
    )


def summarize(samples: list[Sample]) -> list[dict]:
    rows = []
    for cap in sorted({s.cap_w for s in samples}, reverse=True):
        got = [s for s in samples if s.cap_w == cap]

        def med(field: str) -> float | None:
            vals = [getattr(s, field) for s in got if getattr(s, field) is not None]
            return round(statistics.median(vals), 3) if vals else None

        rows.append({"cap_w": cap, "n": len(got), "prompt_tok_s": med("prompt_tok_s"),
                     "gen_tok_s": med("gen_tok_s"), "gen_avg_w": med("gen_avg_w"),
                     "prompt_j_per_tok": med("prompt_j_per_tok"), "gen_j_per_tok": med("gen_j_per_tok"),
                     "max_temp_c": max(s.max_temp_c for s in got), "max_sm_mhz": max(s.max_sm_mhz for s in got)})
    top = rows[0] if rows else None
    for r in rows:  # relative to the highest cap: what each cut costs in speed and saves in energy
        for key in ("gen_tok_s", "prompt_tok_s", "gen_j_per_tok"):
            r[f"{key}_vs_top"] = (round(r[key] / top[key], 3)
                                  if top and r[key] is not None and top[key] else None)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu", type=int, required=True, help="nvidia-smi index of the card")
    ap.add_argument("--port", type=int, required=True, help="llama-server port on this host")
    ap.add_argument("--caps", default="250,200,150,100", help="watts, comma separated, run in this order")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=512)  # ~11 s of writing at 46 tok/s
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--idle-w", type=float, default=80.0, help="card must draw under this before a request")
    ap.add_argument("--settle-s", type=float, default=15.0)
    ap.add_argument("--max-retries", type=int, default=6, help="per cap, for samples thrown away as dirty")
    ap.add_argument("--out", default=f"/tmp/gpu-power-bench-{time.strftime('%Y%m%dT%H%M%S')}")
    args = ap.parse_args(argv)

    base = f"http://127.0.0.1:{args.port}"
    caps = [int(c) for c in args.caps.split(",")]
    original, lo, hi = read_limits(args.gpu)
    bad = [c for c in caps if not lo <= c <= hi]
    if bad:
        print(f"caps {bad} outside this card's range {lo:.0f}-{hi:.0f} W", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"card {args.gpu}: limit {original:.0f} W (range {lo:.0f}-{hi:.0f}); results in {out}")
    print(f"if this process is SIGKILLed, restore with: sudo nvidia-smi -i {args.gpu} -pl {original:.0f}")

    def _raise(signum, _frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise)

    sampler = Sampler(args.gpu)
    sampler.start()
    prompt = fixed_prompt()
    samples: list[Sample] = []
    dropped: list[dict] = []
    try:
        for cap in caps:
            set_limit(args.gpu, cap)
            print(f"\n== {cap} W ==", flush=True)
            time.sleep(args.settle_s)
            one_request(base, prompt, 16, args.slot)  # warm-up at this cap, not recorded
            rep = attempts = 0
            while rep < args.repeats and attempts < args.repeats + args.max_retries:
                attempts += 1
                if not wait_until_quiet(base, sampler, args.slot, args.idle_w, timeout_s=180):
                    dropped.append({"cap_w": cap, "why": "never_quiet"})
                    print("  card never went quiet in 180 s; skipping one attempt", flush=True)
                    continue
                watch = SlotWatch(base, args.slot)
                watch.start()
                t_send, _t_recv, timings = one_request(base, prompt, args.n_predict, args.slot)
                watch.stop()
                watch.join()
                time.sleep(0.5)  # let the sampler cover the end of the window
                if watch.dirty:
                    dropped.append({"cap_w": cap, "why": "other_slot_busy"})
                    print("  another request shared the card; sample thrown away", flush=True)
                    continue
                s = to_sample(cap, rep, sampler, t_send, timings)
                samples.append(s)
                rep += 1
                print(f"  rep {rep}: read {s.prompt_tok_s} tok/s, write {s.gen_tok_s} tok/s, "
                      f"{s.gen_avg_w} W, {s.gen_j_per_tok} J/token, {s.max_temp_c} C", flush=True)
    finally:
        try:
            set_limit(args.gpu, original)
            print(f"\nrestored card {args.gpu} to {original:.0f} W")
        except Exception as exc:  # noqa: BLE001
            print(f"\nRESTORE FAILED ({exc}); run: sudo nvidia-smi -i {args.gpu} -pl {original:.0f}",
                  file=sys.stderr)
        sampler.stop()

    rows = summarize(samples)
    with (out / "samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(Sample.__dataclass_fields__))
        w.writeheader()
        w.writerows(asdict(s) for s in samples)
    meta = {"gpu": args.gpu, "port": args.port, "original_limit_w": original, "caps": caps,
            "n_predict": args.n_predict, "prompt_chars": len(prompt), "dropped": dropped,
            "model": http_json(f"{base}/props", timeout=10).get("model_path")}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "by_cap": rows}, indent=2))
    print("\ncap  read tok/s  write tok/s  write W  J/token(write)  vs top cap (speed / energy)  max C")
    for r in rows:
        print(f"{r['cap_w']:>4} {r['prompt_tok_s']!s:>10} {r['gen_tok_s']!s:>12} {r['gen_avg_w']!s:>8} "
              f"{r['gen_j_per_tok']!s:>15}  {r['gen_tok_s_vs_top']!s:>8} / {r['gen_j_per_tok_vs_top']!s:<8}"
              f"          {r['max_temp_c']}")
    print(f"\nwritten: {out / 'samples.csv'}, {out / 'summary.json'}; dropped samples: {len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
