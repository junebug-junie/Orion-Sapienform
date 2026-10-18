"""Offline P0 timing/replay evidence. python -m orion.retro.preflight --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

from .probe import CoreProbe


def benchmark(core, frames, replay_frames, *, clock=time.perf_counter, sleep=time.sleep):
    if frames < 1 or replay_frames < 1:
        raise ValueError("frame counts must be positive")
    fps = core.av.timing.fps
    period = 1 / fps
    for _ in range(120):
        core.step()  # warm-up excluded
    video_before, audio_before = core.video_frames, core.audio_samples
    start = clock()
    durations, misses = [], 0
    for frame in range(frames):
        frame_start = clock()
        core.step()
        end = clock()
        durations.append(end - frame_start)
        deadline = start + (frame + 1) * period
        misses += end > deadline
        sleep(max(0, deadline - end))
    elapsed = clock() - start
    video = core.video_frames - video_before
    audio = core.audio_samples - audio_before
    state = core.snapshot()
    checkpoint_video = core.frame_hash
    sequence = [(1 << (i % 12), 1 << ((i + 3) % 12)) for i in range(replay_frames)]
    # The first continuation is uninterrupted: two equally broken restores
    # must never manufacture evidence of correct serialization.
    def trial():
        trace = hashlib.sha256()
        observed = False
        for p1, p2 in sequence:
            value = core.step(p1, p2)
            observed |= value is not None
            trace.update((value or "missing").encode())
        return trace.hexdigest(), hashlib.sha256(core.snapshot()).hexdigest(), observed
    first = trial()
    core.restore(state)
    # NULL video callbacks duplicate the frontend image, not serialized core RAM.
    core.frame_hash = checkpoint_video
    second = trial()
    actual_fps = frames / elapsed if elapsed > 0 else 0
    gates = {
        "nominal_cadence": abs(actual_fps / fps - 1) <= .01,
        "deadline_misses_below_one_percent": misses / frames < .01,
        "video_observed": video > 0,
        "audio_observed": audio > 0,
        "same_core_replay": first == second and first[2],
        "thirty_minute_sample": frames / fps >= 1800,
        "replay_1000_frames": replay_frames >= 1000,
    }
    return {
        "schema_version": "retro.preflight.v1",
        "status": "PASS" if all(gates.values()) else "INCOMPLETE_OR_FAILED",
        "identity": core.identity, "frames": frames, "replay_frames": replay_frames,
        "nominal_fps": fps, "elapsed_seconds": elapsed, "actual_fps": actual_fps,
        "step_p95_seconds": sorted(durations)[max(0, int(len(durations) * .95) - 1)],
        "step_median_seconds": statistics.median(durations),
        "deadline_misses": misses, "video_callbacks": video,
        "audio_sample_frames": audio, "sample_rate": core.av.timing.sample_rate,
        "replay_checksums": [first[:2], second[:2]], "gates": gates,
        "unsupported_environment_commands": sorted(core.unsupported),
        "unverified": ["sandbox", "co_op_port_effects", "browser_audio_video",
                       "human_latency", "policy_competence", "cognition", "host_load_matrix"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument("--core-sha256", required=True, help="expected trusted core build digest")
    parser.add_argument("--rom", required=True, type=Path)
    parser.add_argument("--rom-sha256", required=True, help="expected operator-owned ROM digest")
    parser.add_argument("--frames", type=int, default=110000)
    parser.add_argument("--replay-frames", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.frames <= 1000000 or not 1 <= args.replay_frames <= 100000:
        parser.error("frames must be 1..1000000; replay frames 1..100000")
    from .probe import digest
    for path, expected in ((args.core, args.core_sha256), (args.rom, args.rom_sha256)):
        if digest(path) != expected:
            parser.error("input SHA-256 mismatch")
    # Refuse to overwrite evidence. Open before starting the long native run.
    with args.output.open("x") as output:
        core = None
        try:
            core = CoreProbe(args.core, args.rom)
            result = benchmark(core, args.frames, args.replay_frames)
        except Exception as exc:
            result = {"schema_version": "retro.preflight.v1", "status": "ERROR",
                      "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            if core is not None:
                core.close()
        result["host"] = {"machine": platform.machine(), "cpu_count": os.cpu_count(),
                          "load_average": os.getloadavg(),
                          "affinity": sorted(os.sched_getaffinity(0))}
        json.dump(result, output, indent=2)
        output.write("\n")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
