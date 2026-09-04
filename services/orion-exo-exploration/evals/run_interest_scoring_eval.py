"""Eval: precision/recall for `app.crawl.interest.score_candidate` against
real KSL titles.

Every title below is real -- captured live 2026-09-04 from
https://classifieds.ksl.com/search/cat/Electronics, .../Computers, and
.../FREE (the same fetch that produced tests/fixtures/ksl_category_sample.html).
Labels are a human judgment of "is this genuinely a tech/compute item,"
independent of whether our fixed keyword list happens to catch it -- so this
number is an honest measure of the rule set's real recall, not a number
tuned to make the rule set look good. Two labeled positives ("LOADED HP
ELITEBOOK ...", the Alienware ultrawide monitor) are known misses: the
Elitebook's title carries no literal keyword from the seed list, and the
Alienware title says "Ultrawide QHD", not the exact phrase "ultrawide
monitor" the rules match on. Both are left in deliberately, per AGENTS.md's
metric-quality-gate: a rule set that only ever gets graded on cases it
already gets right is not a real precision/recall check.

Deterministic, no DB, no network, no LLM call -- matches
services/orion-cortex-exec/evals/run_current_turn_signal_eval.py's shape.

Run: .venv/bin/python services/orion-exo-exploration/evals/run_interest_scoring_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
for _path in (_SERVICE_DIR, _REPO_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.crawl.interest import InterestRule, rules_from_rows, score_candidate  # noqa: E402
from app.storage.repository import _SEED_KEYWORDS  # noqa: E402

_RULES = [InterestRule(keyword=k, weight=1.0) for k in _SEED_KEYWORDS]

# (title, price, is_tech_compute_ground_truth)
_FIXTURES: list[tuple[str, float | None, bool]] = [
    # --- Electronics (real, 2026-09-04) ---
    ("GX1400 TRANSCEIVER", 100.0, False),
    (
        "We buy Cracked phones! Get cash Now ! iphone 11-17, Samsung galaxy S10-25, "
        "Google pixel 6-10, ipads and apple watches and more",
        999.0,
        False,
    ),
    (
        "Today's deal on iphone 11, 11 Pro, 11 Pro max, Samsung galaxy, google pixels "
        "and many more. Prices in description",
        165.0,
        False,
    ),
    ("James Bond 60th Anniversary Pinball!", 17500.0, False),
    ("Iphone 12 Pro Max - 256GB - Unlocked", 449.0, False),
    ("Desk Lamp", 10.0, False),
    ("Antique Radio Shack Meter", 10.0, False),
    ("Phone Soap", 10.0, False),
    ("Dish TV Receiver", 10.0, False),
    ("Iphone 12 64gb - Unlocked", 299.0, False),
    ("White T smart Projector with WiFi and Bluetooth, Stand and Remote Control", 60.0, False),
    # --- Computers (real, 2026-09-04) ---
    ("Gen3 Starlink Mobile Satellite Internet Rental - Unlimited Data", None, False),
    ('Apple iMac 27" 5K 2020 i5 1TB SSD 32GB Tahoe', 560.0, True),
    ("Brand new Lenovo ThinkVision 31.5 inch Monitor (P32p-20)", 200.0, False),
    ("LOADED HP ELITEBOOK  13TH GEN I7 32GB 512GB  WIN 11 W/FACTORY WARRANTY", 450.0, True),
    ("Voxelab 3D Printer", 80.0, False),
    ("Alienware AW3425DW 34in QD-OLED 240Hz Ultrawide QHD 3440x1440p", 400.0, True),
    ("Asus Rog Nuc 2025 - RTX 5060 - 32 GB DDR5", 2900.0, True),
    ('Dynabook Portege X40-M Laptop, 14" FHD, Ultra 7 155H, 32GB, 1TB SSD', 851.0, True),
    (
        "Dell Optiplex i5-10400 Windows 11 Desktop Computer 256GB SSD 16GB RAM Wifi and Bluetooth",
        240.0,
        True,
    ),
    ("M2 Apple Vision Pro with extras", 2000.0, False),
    ("HP Printer OfficeJetPro 8128e New in box", 100.0, False),
    # --- FREE (real, 2026-09-04) ---
    ("Snow Storm Sleds", 0.0, False),
    ("Blue Bodyboard with Leash", 0.0, False),
    ("Kids Compound Bows with Arrows and Camo Carrying Cases", 0.0, False),
    ("bed set with under bed storage and side pier cabinet", 0.0, False),
    ("Box Of Firewood", 0.0, False),
    ("FREE Assorted Tile Samples Mosaic Hexagon Chevron Patterns", 0.0, False),
    ("FREE Riding Mower", 0.0, False),
    ("Free dresser", 0.0, False),
    ("FREE Love Seat Couches", 0.0, False),
    ("Free California king mattress", 0.0, False),
    ("free roo", 0.0, False),
]

# Precision floor: a false positive here means the rule set is flagging
# obviously irrelevant listings as tech/compute -- should almost never
# happen given a fixed, specific keyword list.
_PRECISION_FLOOR = 0.9
# Recall floor: a fixed keyword list will always miss some real listings
# (see the two documented misses above) -- 0.5 is a floor, not a target.
_RECALL_FLOOR = 0.5


def run() -> int:
    print("\n=== interest scoring precision/recall eval (real KSL titles, 2026-09-04) ===")
    tp = fp = fn = tn = 0
    for title, price, is_positive in _FIXTURES:
        score, reasons = score_candidate(title=title, description=None, price=price, rules=_RULES)
        predicted_positive = score > 0
        if predicted_positive and is_positive:
            tp += 1
            status = "TP"
        elif predicted_positive and not is_positive:
            fp += 1
            status = "FP"
        elif not predicted_positive and is_positive:
            fn += 1
            status = "FN"
        else:
            tn += 1
            status = "TN"
        reason_str = "; ".join(reasons) if reasons else "(no rule fired)"
        print(f"  [{status}] score={score:.1f} title={title[:60]!r} reasons={reason_str}")

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"\nRESULT: tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"precision={precision:.3f} (floor {_PRECISION_FLOOR})")
    print(f"recall={recall:.3f} (floor {_RECALL_FLOOR})")

    ok = precision >= _PRECISION_FLOOR and recall >= _RECALL_FLOOR
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
