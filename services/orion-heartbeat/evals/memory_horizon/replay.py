# Heartbeat memory-horizon replay: see README.md in this directory.
import csv, sys, json, random
from app.substrate.ensemble import EnsembleSubstrate, EnsembleConfig
from app.substrate.reconstruction import compute_h1_ensemble
from app.substrate.routing import route_atom
arm, seed = sys.argv[1], int(sys.argv[2])
rows=[r for r in csv.reader(open('/w/atoms.csv'))]
def f(x,d):
    try: return float(x)
    except: return d
ev=[(float(r[0]),r[1],r[2],f(r[3],None),f(r[4],None),f(r[5],None)) for r in rows]
rng=random.Random(7)
if arm=='organ_shuffle':
    org=[e[1] for e in ev]; rng.shuffle(org); ev=[(e[0],o)+e[2:] for e,o in zip(ev,org)]
elif arm=='order_shuffle':
    ts=[e[0] for e in ev]; body=[e[1:] for e in ev]; rng.shuffle(body); ev=[(t,)+b for t,b in zip(ts,body)]
ens=EnsembleSubstrate(config=EnsembleConfig(n_trajectories=8), base_seed=seed)
REHEAT=0.0076; t0=ev[0][0]; next_dr=t0+2; next_h1=t0+30
out=open(f'/w/out_{arm}_{seed}.jsonl','w')
for t,src,at,c,s,u in ev:
    while t>=next_dr:
        ens.decay_reheat_tick(REHEAT); next_dr+=2
        if next_dr>next_h1:
            h=compute_h1_ensemble(ens); out.write(json.dumps({"t":next_h1,"mean":h.mean_ratio,"std":h.std_ratio,"bulk":h.bulk_penetration_depth,"v":h.verdict})+"\n"); next_h1+=30
    try: ens.absorb(route_atom(source_service=src,atom_type=at,confidence=c,salience=s,uncertainty=u))
    except Exception: pass
out.close()
