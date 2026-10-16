#!/usr/bin/env python3
"""Replay world-first attention winners under alternative rules for continuously-reading body signals.

Read-only analysis for docs/superpowers/specs/2026-10-16-attention-continuous-signal-quota-design.md.

Input: one TSV of (generated_at, world_first trace JSON) rows from substrate_attention_broadcast_log:

    docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc \
      "copy (select generated_at, projection_json->'frame'->'debug'->'world_first' \
             from substrate_attention_broadcast_log where generated_at >= now()-interval '7 days' \
             order by generated_at) to stdout" > /tmp/frames.tsv
    python3 scripts/analysis/replay_attention_continuous_quota.py /tmp/frames.tsv

Every frame's trace already carries each candidate's verdict (eligible, band, score, rank_score),
so the replay re-picks the winner without recomputing any percentile. The status-quo pick is
checked against the recorded winner first.
"""
import json
from datetime import datetime


def load(path):
    out = []
    for line in open(path):
        ts, js = line.rstrip("\n").split("\t", 1)
        js = js.replace("\\\\", "\\")
        try:
            wf = json.loads(js)
        except Exception:
            continue
        if not isinstance(wf, dict):
            continue
        out.append((datetime.fromisoformat(ts.replace(" ", "T")), wf))
    return out


import sys, collections, statistics as st


fr=load(sys.argv[1])
BIO,BUS,PER="node:substrate.biometrics","node:substrate.bus_synaptic","node:substrate.perception"
CONT_INTERNAL={BIO,BUS}
DWELL,WIN,CALM=300.0,600.0,600.0
gaps=[(b[0]-a[0]).total_seconds() for a,b in zip(fr,fr[1:])]
print("frames",len(fr),"span",fr[0][0],"->",fr[-1][0],"median gap s",round(st.median(gaps),1))
def key(c): return c["rank_score"] if c["rank_score"] is not None else (c["score"] or 0.0)
def pick(cands):
    el=[c for c in cands if c]
    if not el: return None
    return max(el,key=lambda c:(key(c), c["source_id"]))["source_id"]
# sanity: reconstruct SQ winner
agree=sum(1 for t,wf in fr if pick([c for c in wf["candidates"] if c["eligible"]])==wf.get("winner"))
print("SQ reconstruction agrees with recorded winner",round(agree/len(fr),4))
# per-source hot-run lengths (seconds, time-weighted by frame gaps)
def runs(src):
    out=[];cur=0.0
    for i,(t,wf) in enumerate(fr):
        c=next((c for c in wf["candidates"] if c["source_id"]==src),None)
        g=gaps[i] if i<len(gaps) else 0
        if c and c["eligible"]: cur+=min(g,120)
        elif cur: out.append(cur);cur=0.0
    return out
for src in [BIO,BUS,PER,"node:substrate.cabinet","node:substrate.execution","node:substrate.codebase","node:substrate.chat","node:substrate.route","world:chat"]:
    r=runs(src)
    if r: print(f"hot runs {src:32s} n={len(r):4d} median={st.median(r):6.0f}s p90={sorted(r)[int(.9*(len(r)-1))]:6.0f}s  >300s:{sum(1 for x in r if x>300)/len(r):.2f}")
# options
def simulate(name, rule, sources):
    hot_hist=collections.defaultdict(collections.deque)   # src -> deque of (epoch, dur) hot
    calm_since={}; open_={}
    wins=collections.Counter(); instead=collections.Counter(); examples=[]
    for i,(t,wf) in enumerate(fr):
        e=t.timestamp(); g=min(gaps[i] if i<len(gaps) else 44.0,120.0)
        cands=[]
        for c in wf["candidates"]:
            s=c["source_id"]
            if s in sources:
                ok=rule(s,c,e,g,hot_hist,calm_since,open_)
                if ok: cands.append(c)
            elif c["eligible"]: cands.append(c)
        w=pick(cands); wins[w]+=1
        sq=wf.get("winner")
        if sq in sources and w!=sq:
            instead[w]+=1
            if w is not None and len(examples)<4: examples.append((t.strftime("%m-%d %H:%M"),sq.split(".")[-1],w.split(".")[-1] if "." in w else w))
    n=len(fr)
    print(f"\n== {name}: no-winner {wins[None]/n:.3f}")
    for s,k in wins.most_common():
        if s: print(f"   {s:32s} {k:5d} {k/n:.3f}")
    tot=sum(instead.values())
    print("   ticks that SQ gave to an affected source now go to:",{(k or 'NO WINNER'):round(v/tot,3) for k,v in instead.most_common()} if tot else {}, "examples",examples)
    return wins
SQ=simulate("status quo (recomputed)", lambda s,c,*a: c["eligible"], set())
def rule_a(s,c,*a): return c["eligible"] and c["band"]=="unusual"
simulate("(a) continuous internal need 'unusual' (top 1%)", rule_a, CONT_INTERNAL)
simulate("(a+) also perception", rule_a, CONT_INTERNAL|{PER})
def rule_b(s,c,e,g,hh,cs,op):
    d=hh[s]
    if c["eligible"]: d.append((e,g))
    while d and d[0][0] <= e-WIN: d.popleft()
    return c["eligible"] and sum(x[1] for x in d) > DWELL
simulate("(b) hot >300s within trailing 600s (episode dwell), still hot now", rule_b, CONT_INTERNAL)
simulate("(b+) also perception", rule_b, CONT_INTERNAL|{PER})
def rule_c(s,c,e,g,hh,cs,op):
    # self_signal_episode semantics: open on dwell, close after CALM s continuously calm; compete while open
    d=hh[s]
    if c["eligible"]: d.append((e,g)); cs.pop(s,None)
    else: cs.setdefault(s,e)
    while d and d[0][0] <= e-WIN: d.popleft()
    if not op.get(s) and sum(x[1] for x in d) > DWELL: op[s]=True
    if op.get(s) and s in cs and e-cs[s] >= CALM: op[s]=False
    return bool(op.get(s))
simulate("(c) one verdict = metacog self_signal_episode open (dwell in, 600s calm out)", rule_c, CONT_INTERNAL)
