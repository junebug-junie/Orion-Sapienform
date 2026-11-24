"""Game company patch 2: re-run the step 0 look fixture through the code that now ships.

Two parts, both deterministic, no network, no GPU, no database:

1. **Calibration on the fixture (always runs).** Every sequence of
   ``services/orion-hub/evals/fixtures/retro_look_step0.json`` has a screen-change value and a label: scripted
   input (walk, long hold, map, menu) or none (idle, still). Per game, the shipped rule
   (``orion.retro.watch.separating_edge``: at most 10% of no-input values and at least 90% of input values at or
   above one floor) is applied to the look-window values and to the 2-second step values. The live rule also needs
   30 minutes of checks; the fixture has 5 to 22 sequences a game, so this shows only whether one value CAN
   separate them. The spec's own finding to reproduce: The Legend of Zelda (NES) does not separate.

2. **Frames through the shipping decoder (only with the private kit).** When ``~/.orion/retro-watch-fixture/``
   is present and the step 0 frames rendered from its saves are on disk (``~/.orion/retro-spike/raw/<sequence>/``,
   owner-only; never committed), each frame is packed with the ENGINE's own packer
   (services/orion-retro/app/media.py), decoded with ``orion.retro.frames``, and the screen change is computed
   again the way step 0 did. The result is reported beside the fixture's number. Without the kit: SKIPPED, said
   plainly. No model is called: the notes' re-run on the live lane is a separate, leased check.

Run:  PYTHONPATH=<repo>:<repo>/services/orion-hub python services/orion-hub/evals/retro_look_replay.py [--output out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "services/orion-hub/evals/fixtures/retro_look_step0.json"
KIT = Path(os.environ.get("ORION_RETRO_WATCH_KIT", Path.home() / ".orion/retro-watch-fixture"))
RAW = Path(os.environ.get("ORION_RETRO_SPIKE_RAW", Path.home() / ".orion/retro-spike/raw"))
TOLERANCE = 0.002          # step 0 scaled 5/6-bit channels with floor division; the decoder rounds

GAMES = {"crystali_": "crystalis", "the_lege_": "the-legend-of-zelda-prg1"}
NO_INPUT = ("ctl_idle", "ctl_menu")


def label(name: str) -> tuple[str, bool]:
    """(game, scripted input?) of one sequence, from the fixture's own naming (its how_rendered field)."""
    for prefix, game in GAMES.items():
        if name.startswith(prefix):
            return game, "_walk_" in name
    return "a-link-to-the-past", name not in NO_INPUT


def sequences(fixture: dict) -> dict[str, dict]:
    out = {s["sequence"]: s["screen_change"] for s in fixture["sequences"]}
    out.update(fixture["screen_change_only"])
    return out


def calibration_part(fixture: dict) -> dict:
    from orion.retro import watch as W

    result: dict = {}
    seqs = sequences(fixture)
    for game in sorted({label(n)[0] for n in seqs}):
        mine = {n: v for n, v in seqs.items() if label(n)[0] == game}
        row: dict = {"input": sorted(n for n in mine if label(n)[1]), "no_input": sorted(n for n in mine if not label(n)[1])}
        for stat in ("window_mean", "step_median_mean"):
            hi, hn = [0] * W.N_BINS, [0] * W.N_BINS
            for n, v in mine.items():
                (hi if label(n)[1] else hn)[W.bin_of(float(v[stat]))] += 1
            best = W.separating_edge(hi, hn)
            row[stat] = ({"floor": round(W.EDGES[best[0]], 5), "no_input_at_or_above": round(best[1], 3),
                          "input_at_or_above": round(best[2], 3)} if best else {"floor": None})
        result[game] = row
    zelda = result.get("the-legend-of-zelda-prg1", {})
    result["spec_check"] = {
        "zelda_nes_does_not_separate": all(zelda.get(s, {}).get("floor") is None for s in ("window_mean", "step_median_mean")),
        "a_link_to_the_past_separates_on_the_look_window": result.get("a-link-to-the-past", {}).get("window_mean", {}).get("floor") is not None,
    }
    return result


def engine_media():
    spec = importlib.util.spec_from_file_location("orion_retro_engine_media", REPO / "services/orion-retro/app/media.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pack(media, raw: bytes, w: int, h: int, fmt: int) -> bytes:
    probe = SimpleNamespace(video=raw, pcm=bytearray(), geometry=(w, h), pixel_format=fmt,
                            av=SimpleNamespace(timing=SimpleNamespace(sample_rate=32040.0)), audio_samples=0)
    sink = []
    media.MediaProbe.publish(probe, SimpleNamespace(put_nowait=sink.append), 2)
    return media.compress(sink[0])


def frames_part(fixture: dict) -> dict:
    if not KIT.is_dir():
        return {"status": "SKIPPED", "why": f"the private kit {KIT} is not on this host"}
    if not RAW.is_dir():
        return {"status": "SKIPPED", "why": f"the kit is here, but no rendered frames in {RAW}; render them with the "
                                            "kit's scripts/render.py inside the orion-retro image"}
    import numpy as np

    from orion.retro.frames import decode_packet, luma_grid, screen_change

    media = engine_media()
    seqs = sequences(fixture)
    rows, close, total = {}, 0, 0
    for name, want in sorted(seqs.items()):
        d = RAW / name
        if not (d / "meta.json").is_file():
            continue
        meta = json.loads((d / "meta.json").read_text())
        w, h, fmt = int(meta["w"]), int(meta["h"]), int(meta.get("fmt", 2))
        grids = [luma_grid(decode_packet(pack(media, (d / f"f{c['n']:02d}.raw").read_bytes(), w, h, fmt)).rgb)
                 for c in meta["caps"]]
        n = len(grids)
        first = n - 16 if n >= 31 else 0                 # step 0: the last 30 s is the look window
        prev = grids[first - 1] if first > 0 else grids[0]
        window = max(screen_change(g, prev) for g in grids[first:])
        steps = [screen_change(grids[i], grids[i - 1]) for i in range(1, n)]
        step_median = float(np.median(steps)) if steps else 0.0
        ok = abs(window - want["window_mean"]) <= TOLERANCE and abs(step_median - want["step_median_mean"]) <= TOLERANCE
        rows[name] = {"window_mean": round(window, 4), "fixture_window_mean": want["window_mean"],
                      "step_median_mean": round(step_median, 4), "fixture_step_median_mean": want["step_median_mean"],
                      "within_tolerance": ok}
        close += ok
        total += 1
    if total == 0:
        return {"status": "SKIPPED", "why": f"no sequence of the fixture has frames in {RAW}"}
    return {"status": "RAN", "sequences": total, "within_tolerance": close, "tolerance": TOLERANCE, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, default=None, help="write the full result as JSON here")
    args = ap.parse_args(argv)
    fixture = json.loads(FIXTURE.read_text())
    calibration = calibration_part(fixture)
    frames = frames_part(fixture)
    print("Part 1, calibration rule on the fixture's screen-change values:")
    for game, row in calibration.items():
        if game == "spec_check":
            continue
        print(f"  {game}: {len(row['input'])} input / {len(row['no_input'])} no-input sequences; "
              f"look window floor {row['window_mean']['floor']}, 2 s step floor {row['step_median_mean']['floor']}")
    for k, v in calibration["spec_check"].items():
        print(f"  {k}: {v}")
    print("Part 2, frames through the shipping decoder:")
    if frames["status"] == "RAN":
        print(f"  RAN: {frames['within_tolerance']} of {frames['sequences']} sequences reproduce the fixture's "
              f"numbers within {TOLERANCE}")
        for name, r in frames["rows"].items():
            if not r["within_tolerance"]:
                print(f"    differs: {name} {r}")
    else:
        print(f"  {frames['status']}: {frames['why']}")
    headline = fixture["headline"]["chat_lane_model_first_prompt_2x"]
    print(f"Fixture headline beside it (not re-run here: needs the live chat lane): {headline['place_and_action_right']} "
          f"of {headline['notes']} notes with place and action right, {headline['notes_with_a_wrong_detail']} with a wrong "
          "detail.")
    out = {"calibration": calibration, "frames": frames, "fixture_headline": fixture["headline"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1))
    ok = calibration["spec_check"]["zelda_nes_does_not_separate"] and (
        frames["status"] != "RAN" or frames["within_tolerance"] == frames["sequences"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
