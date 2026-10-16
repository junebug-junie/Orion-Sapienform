"""A seeded synthetic walkway stream for the body-match rollback test.

People walk across the frame in passings, several at once, with body vectors
near their own base vector (noise sized so many scores sit near 0.80) and an
occasional face. Plain data only: ``build(vi)`` turns it into ``vi.CropRow``
for whichever ``vision_individuals`` module is passed in, so the same stream
can be run through the code before and after a change.

The golden digest in test_vision_body_match.py was produced by running this
stream through the pre-continuation ``apply_batch`` (main at f2bccf3dc).
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

T0 = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc)
DIM = 32


def stream(seed: int = 7, people: int = 24, passings: int = 90):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(people, DIM))
    shared = rng.normal(size=DIM)
    base = base + 1.2 * shared            # different people score ~0.5-0.8 against each other
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    faces = rng.normal(size=(people, 16))
    faces /= np.linalg.norm(faces, axis=1, keepdims=True)
    rows = []
    frame = 0
    for k in range(passings):
        start = T0 + timedelta(seconds=int(rng.integers(0, 6 * 3600)))
        who = rng.choice(people, size=int(rng.integers(1, 4)), replace=False)
        y = float(rng.uniform(300, 1500))
        for step in range(int(rng.integers(2, 9))):
            t = start + timedelta(seconds=2.6 * step)
            obs = f"o{frame:05d}"
            frame += 1
            for slot, p in enumerate(who):
                h = float(rng.uniform(160, 350))
                x = 200 + 180 * step + 400 * slot + float(rng.normal(0, 30))
                e = base[p] + rng.normal(scale=0.09, size=DIM)
                f = None
                if rng.random() < 0.15:
                    f = faces[p] + rng.normal(scale=0.35, size=16)
                rows.append(dict(
                    crop_id=f"cropobs:{obs}:{slot}", observation_id=obs, observed_at=t,
                    landed=t + timedelta(seconds=float(rng.uniform(1, 40))),
                    box=(x, y, x + h * 0.4, y + h), embedding=tuple(float(v) for v in e),
                    face=None if f is None else tuple(float(v) for v in f)))
    rows.sort(key=lambda r: (r["landed"], r["crop_id"]))
    return rows


def build(vi, rows):
    return [vi.CropRow(
        crop_id=r["crop_id"], observation_id=r["observation_id"], stream_id="walkway",
        observed_at=r["observed_at"], label="person", box_xyxy=r["box"], zone="walkway",
        embedding=r["embedding"], embedding_ref=None, artifact_id=None,
        face_embedding=r["face"]) for r in rows]


def run_batches(vi, rows, tz, **kwargs):
    """apply_batch per 60 s landing bucket, like the live loop. Returns
    crop_id -> individual id (deterministic ids)."""
    crops = build(vi, rows)
    batches = defaultdict(list)
    for c, r in zip(crops, rows):
        batches[int(r["landed"].timestamp() // 60)].append(c)
    inds, sightings, member = {}, {}, {}
    counter = itertools.count()
    for key in sorted(batches):
        b = batches[key]
        lo = min(c.observed_at for c in b) - timedelta(seconds=60)
        hi = max(c.observed_at for c in b) + timedelta(seconds=60)
        nearby = defaultdict(list)
        for s in sightings.values():
            if s.ended_at >= lo and s.started_at <= hi:
                nearby[s.individual_id].append(s)
        res = vi.apply_batch(b, individuals=inds, latest_sightings={}, nearby_sightings=dict(nearby),
                             no_embed_zones=(), merge_gap_sec=60.0, tz=tz,
                             new_id=lambda: f"i{next(counter):05d}", **kwargs)
        inds = res.individuals
        sightings.update(res.sightings)
        member.update(res.crop_individual)
    return member, inds


def digest(member, inds) -> str:
    h = hashlib.sha256()
    for cid in sorted(member):
        h.update(f"{cid}={member[cid]};".encode())
    for iid in sorted(inds):
        ind = inds[iid]
        h.update(f"{iid}:{ind.centroid_n}:{ind.face_n}:{ind.sighting_count}:"
                 f"{','.join(f'{v:.6f}' for v in ind.centroid)};".encode())
    return h.hexdigest()
