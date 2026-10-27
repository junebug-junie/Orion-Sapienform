"""Train the action-conditioned TransitionModel on dojo transitions and gate it against counts.

Held out by dojo worker (seed), never by row: rows from one worker share a trajectory archive.
Gate per head: held-out log loss against a count baseline (Beta(1,1) rate per skill x side open x
threat band, the same backoff shape as orion-prediction's action_beta_backoff), with a paired
bootstrap over held-out rows. PASS only when the 95% interval of (baseline - model) is above 0.

  python services/orion-world-model/scripts/train_transition.py --data <dojo run dir> --out <dir>
Weights go to <out>/transition.pt (never committed); the report to <out>/report.json.
"""
import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.transition import HEADS, TransitionModel, loss, zelda_features, zelda_targets  # noqa: E402


def load(data):
    rows = []
    for f in sorted(glob.glob(str(Path(data) / '**' / 'transitions.jsonl'), recursive=True)):
        for line in open(f):
            r = json.loads(line)
            y = zelda_targets(r)
            if y is not None:
                rows.append((r['seed'], zelda_features(r['before'], r['goal'], r.get('open_sides')), y, r))
    return rows


def count_key(x):
    skill = x[:6].index(1.0) if 1.0 in x[:6] else -1
    side_open = int(any(a and b for a, b in zip(x[6:10], x[10:14])))
    threat = 0 if x[16] == 0 else (1 if x[17] > 0.2 else 2)
    return skill, side_open, threat


def count_baseline(train):
    stats = defaultdict(lambda: [[1.0, 2.0] for _ in HEADS])   # Beta(1,1): successes+1, trials+2
    for _, x, y, _ in train:
        for i in range(len(HEADS)):
            s = stats[count_key(x)][i]; s[0] += y[i]; s[1] += 1
    return lambda x: [s[0] / s[1] for s in stats[count_key(x)]]


def nll(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def main(args):
    torch.manual_seed(0); rng = random.Random(0)
    rows = load(args.data)
    seeds = sorted({r[0] for r in rows})
    held = set(seeds[::4]) if len(seeds) >= 4 else {seeds[-1]}
    train = [r for r in rows if r[0] not in held]; test = [r for r in rows if r[0] in held]
    if len(seeds) < 2 or not train or not test:
        sys.exit(f'need transitions from at least 2 dojo workers (got seeds={seeds}); no verdict')
    X = torch.tensor([r[1] for r in train]); Y = torch.tensor([r[2] for r in train])
    model = TransitionModel()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); loss(model, X[b], Y[b]).backward(); opt.step()
    model.eval()
    base = count_baseline(train)
    with torch.no_grad():
        pred = model.predict(torch.tensor([r[1] for r in test]))
    report = dict(rows=len(rows), train=len(train), test=len(test), held_out_seeds=sorted(held), heads={})
    for i, head in enumerate(HEADS):
        m = [nll(float(pred[head][j]), r[2][i]) for j, r in enumerate(test)]
        b = [nll(base(r[1])[i], r[2][i]) for r in test]
        diffs = [bb - mm for bb, mm in zip(b, m)]
        boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(1000))
        lo, hi = boots[25], boots[974]
        report['heads'][head] = dict(rate=round(sum(r[2][i] for r in test) / max(1, len(test)), 3),
                                     model_nll=round(sum(m) / len(m), 4), count_nll=round(sum(b) / len(b), 4),
                                     gain_ci95=[round(lo, 4), round(hi, 4)], verdict='PASS' if lo > 0 else 'FAIL')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'transition.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=60)
    main(p.parse_args())
