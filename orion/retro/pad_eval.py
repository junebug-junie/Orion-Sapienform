"""Real-core eval specific to the authored Pad Lab ROM; never arbitrary game RAM."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

from .pad_lab import build_rom
from .probe import CoreProbe, digest


class PadProbe(CoreProbe):
    def __init__(self, *args):
        self.panels = None
        self.audio_peak = 0
        super().__init__(*args)
        self.lib.retro_get_memory_size.argtypes = [C.c_uint]
        self.lib.retro_get_memory_size.restype = C.c_size_t
        self.lib.retro_get_memory_data.argtypes = [C.c_uint]
        self.lib.retro_get_memory_data.restype = C.c_void_p

    def _video(self, data, width, height, pitch):
        super()._video(data, width, height, pitch)
        if not data or self.error:
            return
        bpp = 4 if self.pixel_format == 1 else 2
        # Interior panel samples avoid overscan and the dividing edge.
        y, left, right = height // 2, width // 4, 3 * width // 4
        self.panels = [C.string_at(data + y * pitch + x * bpp, bpp).hex()
                       for x in (left, right)]

    def _audio(self, left, right):
        super()._audio(left, right)
        self.audio_peak = max(self.audio_peak, abs(left), abs(right))

    def _audio_batch(self, data, frames):
        # These limits guard diagnostics, not an untrusted native-core boundary.
        if frames > 8192 or (frames and not data):
            self.error = "invalid PCM batch"
            return 0
        self.audio_peak = max(self.audio_peak, max((abs(data[i]) for i in range(frames * 2)), default=0))
        return super()._audio_batch(data, frames)

    def counters(self):
        # RETRO_MEMORY_SYSTEM_RAM=2; Pad Lab owns $02/$03 as per-port counters.
        if self.lib.retro_get_memory_size(2) < 4:
            raise RuntimeError("core exposes insufficient Pad Lab RAM")
        pointer = self.lib.retro_get_memory_data(2)
        if not pointer:
            raise RuntimeError("core exposes no Pad Lab RAM")
        return list(C.string_at(pointer + 2, 2))


def evaluate(core):
    for _ in range(120):
        core.step()
    checkpoint = core.snapshot()
    scenarios = {}
    for name, masks in (("neutral", (0, 0)), ("p1", (1 << 8, 0)),
                        ("p2", (0, 1 << 8)), ("both", (1 << 8, 1 << 8))):
        core.restore(checkpoint)
        core.audio_peak = 0
        for _ in range(7):
            core.step(*masks)
        scenarios[name] = {"counters": core.counters(), "panels": core.panels,
                           "audio_peak": core.audio_peak}
    neutral, p1, p2, both = (scenarios[name] for name in ("neutral", "p1", "p2", "both"))
    checks = {
        "neutral_counters": neutral["counters"] == [0, 0],
        "p1_only": 0 < p1["counters"][0] <= 7 and p1["counters"][1] == 0,
        "p2_only": p2["counters"][0] == 0 and 0 < p2["counters"][1] <= 7,
        "both": all(0 < value <= 7 for value in both["counters"]),
        "video_present": all(row["panels"] is not None for row in scenarios.values()),
        "nonzero_pcm": all(row["audio_peak"] > 0 for row in scenarios.values()),
    }
    if checks["video_present"]:
        checks["p1_changes_left_only"] = (p1["panels"][0] != neutral["panels"][0]
                                          and p1["panels"][1] == neutral["panels"][1])
        checks["p2_changes_right_only"] = (p2["panels"][1] != neutral["panels"][1]
                                           and p2["panels"][0] == neutral["panels"][0])
    return {"schema_version": "retro.pad_lab.eval.v1", "identity": core.identity,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "scenarios": scenarios,
            "scope": "authored NES diagnostic only; no commercial-game competence or browser audio claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", required=True, type=Path)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--rom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if digest(args.rom) != hashlib.sha256(build_rom()).hexdigest():
        parser.error("this eval only accepts the exact authored Pad Lab ROM")
    if digest(args.core) != args.core_sha256:
        parser.error("core fingerprint mismatch")
    with args.output.open("x") as output:
        core = None
        try:
            core = PadProbe(args.core, args.rom)
            result = evaluate(core)
        except Exception as exc:
            result = {"status": "ERROR", "error": str(exc)}
        finally:
            if core is not None:
                core.close()
        json.dump(result, output, indent=2)
        output.write("\n")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
