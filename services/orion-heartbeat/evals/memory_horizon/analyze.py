# Heartbeat memory-horizon replay: see README.md in this directory.
import json, csv, numpy as np
arms=['real_42','real_142','real_242','organ_shuffle_42','order_shuffle_42']
D={a:[json.loads(l) for l in open(f'/w/out_{a}.jsonl')] for a in arms}
T=np.array([d['t'] for d in D['real_42']])
def col(a,k): return np.array([d[k] for d in D[a]])
print("arm                 mean_ratio(avg,sd,min,max)      std_ratio avg  verdicts")
for a in arms:
    m=col(a,'mean'); s=col(a,'std'); v=[d['v'] for d in D[a]]
    vc={x:v.count(x) for x in sorted(set(v))}
    print(f"{a:18s} {m.mean():.3f} {m.std():.3f} {m.min():.3f} {m.max():.3f}   {s.mean():.4f}   {vc}")
def corr(x,y): return float(np.corrcoef(x,y)[0,1])
def agree(a,b): va=[d['v'] for d in D[a]]; vb=[d['v'] for d in D[b]]; return sum(x==y for x,y in zip(va,vb))/len(va)
print("\ncorrelation with real_42 (mean_ratio / std_ratio / bulk) and verdict agreement:")
for a in arms[1:]:
    print(f"  {a:18s} r_mean={corr(col('real_42','mean'),col(a,'mean')):+.2f} r_std={corr(col('real_42','std'),col(a,'std')):+.2f} r_bulk={corr(col('real_42','bulk'),col(a,'bulk')):+.2f} verdict_agree={agree('real_42',a):.0%}")
# seed-mean as signal estimate
seedmean=np.mean([col(a,'mean') for a in arms[:3]],axis=0)
# leaky per-organ counts
ev=[(float(r[0]),r[1]) for r in csv.reader(open('/w/atoms.csv'))]
orgs=['orion-biometrics','orion-bus','orion-cortex-exec','orion-hub']
for tau in (60,300,900):
    X=[]
    for t in T:
        row=[]
        for o in orgs:
            row.append(sum(np.exp(-(t-te)/tau) for te,oo in ev if oo==o and te<=t and t-te<6*tau))
        X.append(row+[1.0])
    X=np.array(X)
    for name,y in (('real_42',col('real_42','mean')),('seed-avg',seedmean)):
        b,res,_,_=np.linalg.lstsq(X,y,rcond=None); pred=X@b; r2=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
        print(f"  leaky counts tau={tau:4d}s -> R^2 on {name:8s} mean_ratio = {r2:.2f}")
# lag-1 autocorr of real_42 mean (is it a smooth state or noise?)
m=col('real_42','mean'); print("\nlag-1 autocorr real_42 mean_ratio:",round(corr(m[:-1],m[1:]),2))
