"""Train the action-conditioned TransitionModel on dojo transitions and gate it against counts.

Rows are deduplicated on (before, goal). Held out by dojo worker (seed); the gate then keeps only
held-out rows whose state (place, coarse tile, hearts) and goal never appear in training, since
every worker starts from the same opening. Gate per head: held-out log loss against a count
baseline shaped like orion-prediction's action_beta_backoff (Beta(1,1) per skill x side open x
threat x health, backing off to skill, then to the global rate, below K rows), with a bootstrap
over (seed, room) clusters. PASS only when the 95% interval of (count - model) is above 0.

  python services/orion-world-model/scripts/train_transition.py --data <dojo run dir> --out <dir>
Weights go to <out>/transition.pt and <out>/transition.npz (never committed); the report to
<out>/report.json. --publish <path> also replaces the numpy weights orion-prediction shadow-scores.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import random
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.transition import FEATURE_VERSION, HEADS, TransitionModel, loss, zelda_features, zelda_targets  # noqa: E402

K = 4   # rows a count key needs before it is trusted over its parent (action_beta_backoff's k)


def state_key(before, goal):
    t = before.get('tile') or [0, 0]
    return (before.get('mode'), str(before.get('level')), before.get('room'), t[0] // 8, t[1] // 6,
            (before.get('hearts') or {}).get('current'), json.dumps(goal, sort_keys=True))


def load(data, keep=1.0):
    """``keep`` < 1: a deterministic sample (by line hash), so a big dojo run trains in minutes."""
    rows, seen = [], set()
    cut = int(keep * 2**32)
    for f in sorted(glob.glob(str(Path(data) / '**' / 'transitions.jsonl'), recursive=True)):
        for line in open(f):
            if keep < 1 and zlib.crc32(line.encode()) >= cut:
                continue
            r = json.loads(line)
            if str(r.get('outcome', '')).startswith('dojo:'):
                continue
            y = zelda_targets(r)
            dup = json.dumps([r['before'], r['goal']], sort_keys=True)
            if y is None or dup in seen:
                continue
            seen.add(dup)
            rows.append(dict(seed=r['seed'], x=zelda_features(r['before'], r['goal'], r.get('open_sides')), y=y,
                             state=state_key(r['before'], r['goal']), cluster=(r['seed'], r['before'].get('room'))))
    return rows


def count_keys(x):
    skill = x[:6].index(1.0) if 1.0 in x[:6] else -1
    side_open = int(any(a and b for a, b in zip(x[6:10], x[10:14])))
    threat = 0 if x[16] == 0 else (1 if x[17] > 0.2 else 2)
    return [(skill, side_open, threat, int(x[15])), (skill,), ()]   # situation -> skill -> global


def count_baseline(train):
    stats = defaultdict(lambda: [[0.0, 0.0] for _ in HEADS])
    for r in train:
        for key in count_keys(r['x']):
            for i in range(len(HEADS)):
                s = stats[key][i]; s[0] += r['y'][i]; s[1] += 1

    def predict(x):
        keys = count_keys(x)
        key = next((k for k in keys if stats[k][0][1] >= K), keys[-1])
        return [(s[0] + 1) / (s[1] + 2) for s in stats[key]]
    return predict


def nll(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def cluster_ci(diffs, clusters, rng, n=1000):
    """95% interval of the mean difference, resampling whole (seed, room) clusters."""
    by = defaultdict(list)
    for d, c in zip(diffs, clusters): by[c].append(d)
    groups = [(sum(g), len(g)) for g in by.values()]
    means = []
    for _ in range(n):
        pick = [rng.choice(groups) for _ in groups]
        means.append(sum(s for s, _ in pick) / sum(c for _, c in pick))
    means.sort()
    return means[int(n * 0.025)], means[int(n * 0.975) - 1], len(groups)


def export(model, path, tag):
    """numpy weights for orion.embodiment.transition.TransitionWeights (orion-prediction has no torch)."""
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items() if k.startswith(('body', 'events'))}
    np.savez(path, feature_version=np.array(FEATURE_VERSION), tag=np.array(tag), **arrays)


def main(args):
    torch.manual_seed(0); rng = random.Random(0)
    torch.set_num_threads(args.threads)   # a CPU-capped container still reports every host core
    rows = load(args.data, args.keep)
    seeds = sorted({r['seed'] for r in rows})
    held = set(seeds[::4]) if len(seeds) >= 4 else set(seeds[-1:])
    train = [r for r in rows if r['seed'] not in held]
    seen = {r['state'] for r in train}
    held_rows = [r for r in rows if r['seed'] in held]
    test = [r for r in held_rows if r['state'] not in seen]
    if len(seeds) < 2 or not train or not test:
        sys.exit(f'need held-out rows from unseen states (seeds={seeds}, test={len(test)}); no verdict')
    X = torch.tensor([r['x'] for r in train]); Y = torch.tensor([r['y'] for r in train])
    model = TransitionModel(n_features=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad(); loss(model, X[b], Y[b]).backward(); opt.step()
    model.eval()
    base = count_baseline(train)
    with torch.no_grad():
        pred = model.predict(torch.tensor([r['x'] for r in test]))
    report = dict(rows_after_dedup=len(rows), train=len(train), held_out=len(held_rows), test_unseen_states=len(test),
                  held_out_share_seen_in_train=round(1 - len(test) / max(1, len(held_rows)), 3),
                  held_out_seeds=sorted(held), heads={})
    for i, head in enumerate(HEADS):
        m = [nll(float(pred[head][j]), r['y'][i]) for j, r in enumerate(test)]
        b = [nll(base(r['x'])[i], r['y'][i]) for r in test]
        lo, hi, n_clusters = cluster_ci([bb - mm for bb, mm in zip(b, m)], [r['cluster'] for r in test], rng)
        report['heads'][head] = dict(rate=round(sum(r['y'][i] for r in test) / len(test), 3),
                                     model_nll=round(sum(m) / len(m), 4), count_nll=round(sum(b) / len(b), 4),
                                     gain_ci95=[round(lo, 4), round(hi, 4)], clusters=n_clusters,
                                     verdict='PASS' if lo > 0 else 'FAIL')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'transition.pt')
    tag = f"{FEATURE_VERSION}:{len(train)}rows:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report['tag'] = tag
    export(model, out / 'transition.npz', tag)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    if args.publish:
        # Shadow scoring in orion-prediction reads this file; replace it atomically.
        dest = Path(args.publish); dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp.npz'); export(model, tmp, tag); os.replace(tmp, dest)
        (dest.parent / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--keep', type=float, default=1.0, help='fraction of rows to sample (deterministic)')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--publish', help='also write the numpy weights here (atomic), e.g. <dojo>/models/transition.npz')
    main(p.parse_args())
