"""Replay the person model's guesses with and without camera priors, read-only.

    POSTGRES_URI=postgresql://... python services/orion-sql-writer/evals/run_office_prior_replay.py

Loads the live album, fits the bars, guesses on the live street exactly as
``run_person_model`` does, but writes nothing. Prints, per camera with a prior,
how many guesses change and to whom. PASS when, on every prior camera, no
guess the scores accepted changes (the prior only fills in unsure guesses),
and at least one guess exists to compare.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from sqlalchemy import create_engine  # noqa: E402

from app import vision_person_model as pm  # noqa: E402
from orion.person_model.camera_priors import load_camera_priors  # noqa: E402


def main() -> int:
    uri = os.environ.get("POSTGRES_URI")
    if not uri:
        print("POSTGRES_URI not set")
        return 2
    cfg = pm.PersonModelConfig()
    tz = ZoneInfo(cfg.local_tz)
    now = datetime.now(timezone.utc)
    priors = load_camera_priors()
    eng = create_engine(uri)
    with eng.connect() as conn:
        album, _ = pm.load_album(conn, tz)
        street, face_street = pm.street_pools(conn, now, cfg, tz)
        not_people = pm.not_a_person_names(conn) if cfg.alpha_split_people_only else frozenset()
        bars = pm.fit_all(album, cfg, street, face_street, not_people)
        blocked = pm._blocked(conn)
    before = {g["individual_id"]: g for g in pm.attribute(street, album, bars, blocked, cfg)}
    after = {g["individual_id"]: g for g in pm.attribute(street, album, bars, blocked, cfg, priors=priors)}
    ok = True
    for stream in sorted(priors):
        ids = [cid for cid, crops in street.items() if pm.cluster_stream(crops) == stream]
        changed = Counter((before[i]["person_label"], after[i]["person_label"]) for i in ids
                          if i in before and i in after and before[i]["person_label"] != after[i]["person_label"])
        accepted_changed = [i for i in ids if i in before and before[i]["band"] == "accept"
                            and after.get(i, {}).get("person_label") != before[i]["person_label"]]
        print(f"{stream}: {len(ids)} unnamed clusters, {sum(1 for i in ids if i in after)} guessed")
        print("  before:", dict(Counter(before[i]["person_label"] for i in ids if i in before)))
        print("  after: ", dict(Counter(after[i]["person_label"] for i in ids if i in after)),
              f"(prior basis: {sum(1 for i in ids if after.get(i, {}).get('basis') == 'prior')})")
        print("  changed:", dict(changed), " accepted guesses changed:", len(accepted_changed))
        ok &= not accepted_changed and any(i in after for i in ids)
    others = [i for i in after if before.get(i, {}).get("person_label") != after[i]["person_label"]
              and pm.cluster_stream(street[i]) not in priors]
    print("guesses changed on cameras without a prior:", len(others))
    ok &= not others
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
