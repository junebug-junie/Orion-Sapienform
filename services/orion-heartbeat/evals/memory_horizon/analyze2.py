# Heartbeat memory-horizon replay: see README.md in this directory.
import json, csv, bisect, numpy as np
D=[json.loads(l) for l in open('/w/out_real_42.jsonl')]
T=np.array([d['t'] for d in D]); y=np.array([d['mean'] for d in D])
ys=np.mean([[json.loads(l)['mean'] for l in open(f'/w/out_real_{s}.jsonl')] for s in (42,142,242)],axis=0)
ev=[(float(r[0]),r[1],r[2]) for r in csv.reader(open('/w/atoms.csv'))]
ts=[e[0] for e in ev]
orgs=['orion-biometrics','orion-bus','orion-cortex-exec','orion-hub']
types=sorted(set(e[2] for e in ev))
def feats(K):
    X=[]
    for t in T:
        i=bisect.bisect_right(ts,t); last=ev[max(0,i-K):i]
        X.append([sum(1 for e in last if e[1]==o)/K for o in orgs]+[sum(1 for e in last if e[2]==a)/K for a in types]+[1.0])
    return np.array(X)
def r2(X,y):
    # 2-fold blocked CV to avoid overfit
    n=len(y); h=n//2; out=[]
    for tr,te in ((slice(0,h),slice(h,n)),(slice(h,n),slice(0,h))):
        b=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0]; p=X[te]@b; out.append(1-((y[te]-p)**2).sum()/((y[te]-y[te].mean())**2).sum())
    return np.mean(out)
print("atom types:",types)
for K in (1,3,10,30,100):
    X=feats(K); print(f"last {K:3d} events (organ+type mix): held-out R^2 seed-avg={r2(X,ys):.2f}")
# seed-to-seed ceiling: how much of one seed is predictable from the others
print("ceiling: R^2 of real_42 from mean of other two seeds:", round(1-((y-np.mean([[json.loads(l)['mean'] for l in open(f'/w/out_real_{s}.jsonl')] for s in (142,242)],axis=0))**2).sum()/((y-y.mean())**2).sum(),2))
