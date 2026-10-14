"""Render linkage decisions as one self-contained HTML review file.

No network: inline CSS and JS, no fonts, images, or fetches. Each decision is a
card with its evidence and per-feature scores, grouped auto-merge / needs your
call / auto-reject, with approve / reject and notes. "Export decisions" builds a
CSV in the browser and downloads it; choices are also kept in localStorage so a
reload does not lose them.
"""

from __future__ import annotations

import html
import json
from typing import Dict, Iterable, List, Optional, Sequence

from orion.schemas.person_model import PersonLinkageDecisionV1

GROUPS = (
    ("auto_merge", "Auto-merge", "Orion would do these on its own: merges above the upper threshold and facet links (a link, never a merge). Each one is logged and can be undone. Flip any you disagree with."),
    ("needs_your_call", "Needs your call", "The possible-match band: the score sits between the two thresholds, or a guard fired. These are the only ones that would become Orion-asks cards in Hub."),
    ("auto_reject", "Auto-reject", "Orion would leave these alone as people: non-matches, topic clusters, leaked prompt fields (flagged, never deleted), and the 'home' alias your answer corrected."),
)


def verdict(d: PersonLinkageDecisionV1) -> str:
    a, c = d.anchor_label, d.candidate_label
    if d.relation == "same_as" and d.group == "auto_merge":
        return f"Same person: “{c}” is “{a}”. Orion would merge them."
    if d.relation == "same_as" and d.rule == "cooccurring_names_guard":
        return f"“{c}” and “{a}” appear side by side in one message, so they may be two people. Same person?"
    if d.relation == "same_as":
        return f"Orion can't tell: is “{c}” the same person as “{a}”?"
    if d.relation == "facet_of":
        return f"“{c}” is something about “{a}”, not the person. Orion would link it, not merge it."
    if d.relation == "junk":
        return f"“{c}” is a prompt field name that leaked into the graph, not a person. Orion would flag it; nothing is deleted."
    if d.relation == "not_home":
        return f"You said home is {d.evidence.get('stated_home', '?')}. This alias calls “{d.evidence.get('place_label', c)}” home. Orion would stop treating it as home (it stays a place you mentioned)."
    if d.relation == "distinct":
        return f"Two different people both called “{c}” appear together. Orion would keep them as separate people. Right?"
    if d.rule == "topic_cluster_concept":
        return f"“{c}” is a topic cluster from the topic modeler, not a person."
    return f"Not the same person as “{a}” (score below the lower threshold)."


def _e(x: object) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _card(d: PersonLinkageDecisionV1, default_choice: str) -> str:
    ev = d.evidence or {}
    rows = []
    for k in ("anchor_node", "candidate_node"):
        node = ev.get(k) or {}
        if node:
            rows.append(
                "<tr><th>{}</th><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                    "anchor" if k == "anchor_node" else "candidate",
                    _e(node.get("node_id")),
                    _e(node.get("label")),
                    _e(" / ".join(str(x) for x in (node.get("kind"), node.get("producer")) if x)),
                )
            )
    snippets = "".join(
        f"<li><span class=src>{_e(s.get('source'))}</span> {_e(s.get('text'))}</li>" for s in (ev.get("snippets") or [])[:5]
    ) or "<li class=muted>No linked memory or chat mentions this label.</li>"
    feats = "".join(
        f"<tr><td>{_e(f.name)}</td><td>{_e(f.level)}</td><td class=num>{f.weight:+.1f}</td><td>{_e(f.detail)}</td></tr>"
        for f in d.features
    )
    score = (
        f"<div class=score>score <b>{d.score:+.1f}</b> &middot; auto-merge at &ge; {d.upper_threshold:+.1f} &middot; auto-reject at &le; {d.lower_threshold:+.1f}</div>"
        if d.score is not None
        else f"<div class=score>decided by rule <code>{_e(d.rule)}</code> (no score)</div>"
    )
    feat_block = f"<details><summary>Per-feature scores</summary><table class=feat><tr><th>feature</th><th>level</th><th>weight</th><th>detail</th></tr>{feats}</table></details>" if feats else ""
    name = _e(d.decision_id)
    return f"""
<article class="card {d.group}" data-id="{name}" data-group="{_e(d.group)}" data-relation="{_e(d.relation)}"
  data-anchor="{_e(d.anchor_id)}" data-anchor-label="{_e(d.anchor_label)}" data-candidate="{_e(d.candidate_id)}"
  data-candidate-label="{_e(d.candidate_label)}" data-score="{'' if d.score is None else d.score}" data-default="{default_choice}">
  <h3>{_e(verdict(d))}</h3>
  {score}
  <table class=nodes>{''.join(rows)}</table>
  <div class=ev><b>Evidence</b><ul>{snippets}</ul></div>
  {feat_block}
  <div class=choice>
    <label><input type=radio name="c-{name}" value=approve> approve</label>
    <label><input type=radio name="c-{name}" value=reject> reject</label>
    <textarea placeholder="notes (optional)" rows=2></textarea>
  </div>
</article>"""


_CSS = """
body{font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#f6f5f2;color:#1d1d1f}
header{background:#1d1d1f;color:#fff;padding:18px 28px;position:sticky;top:0;z-index:2}
header h1{margin:0 0 4px;font-size:20px} header p{margin:2px 0;color:#ccc;font-size:13px}
header button{margin-top:8px;font-size:15px;padding:6px 14px;border-radius:6px;border:0;background:#4c8bf5;color:#fff;cursor:pointer}
main{padding:12px 28px 60px;max-width:1100px}
section>h2{margin:28px 0 4px} section>p{margin:0 0 10px;color:#555}
.card{background:#fff;border-radius:10px;padding:14px 18px;margin:10px 0;border-left:6px solid #999;box-shadow:0 1px 2px #0001}
.card.auto_merge{border-color:#2e9b5f}.card.needs_your_call{border-color:#e0a100}.card.auto_reject{border-color:#9aa0a6}
.card h3{margin:0 0 6px;font-size:16px} .score{color:#444;font-size:13px;margin-bottom:6px}
table{border-collapse:collapse;font-size:13px} td,th{padding:2px 8px 2px 0;text-align:left;vertical-align:top}
.nodes code{font-size:12px;word-break:break-all} .feat td.num{text-align:right;font-variant-numeric:tabular-nums}
.ev ul{margin:4px 0 6px 18px;padding:0} .ev li{margin:2px 0} .src{color:#777;font-size:12px;margin-right:4px}
.muted{color:#888} .choice{margin-top:8px;display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.choice textarea{flex:1;min-width:240px;font:inherit;font-size:13px}
details>summary{cursor:pointer;color:#345} .info td{padding-right:14px}
"""

_JS = r"""
(function(){
  var KEY='person-cleanup-review:'+document.body.dataset.run;
  var saved={}; try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
  var cards=[].slice.call(document.querySelectorAll('article.card'));
  function save(){var o={};cards.forEach(function(c){var r=c.querySelector('input:checked');o[c.dataset.id]={choice:r?r.value:'',notes:c.querySelector('textarea').value}});
    try{localStorage.setItem(KEY,JSON.stringify(o))}catch(e){} count();}
  function count(){var n=0;cards.forEach(function(c){if(c.querySelector('input:checked'))n++});document.getElementById('progress').textContent=n+' of '+cards.length+' cards have a choice';}
  cards.forEach(function(c){var s=saved[c.dataset.id]; var v=s?s.choice:c.dataset.default;
    if(v){var r=c.querySelector('input[value='+v+']'); if(r) r.checked=true;}
    if(s) c.querySelector('textarea').value=s.notes||'';
    c.addEventListener('change',save); c.addEventListener('input',save);});
  count();
  function q(v){v=(v==null?'':String(v));return '"'+v.replace(/"/g,'""')+'"';}
  document.getElementById('export').addEventListener('click',function(){
    var head=['decision_id','group','relation','anchor_id','anchor_label','candidate_id','candidate_label','score','your_choice','notes'];
    var lines=[head.join(',')];
    cards.forEach(function(c){var r=c.querySelector('input:checked');
      lines.push([c.dataset.id,c.dataset.group,c.dataset.relation,c.dataset.anchor,c.dataset.anchorLabel,c.dataset.candidate,c.dataset.candidateLabel,c.dataset.score,r?r.value:'',c.querySelector('textarea').value].map(q).join(','));});
    var blob=new Blob([lines.join('\r\n')+'\r\n'],{type:'text/csv'});
    var a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='person-cleanup-decisions-'+document.body.dataset.run+'.csv';
    document.body.appendChild(a); a.click(); setTimeout(function(){URL.revokeObjectURL(a.href); a.remove();},0);
  });
})();
"""


def render(
    decisions: Sequence[PersonLinkageDecisionV1],
    *,
    title: str,
    run_id: str,
    summary_lines: Iterable[str] = (),
    info_rows: Optional[List[Dict[str, object]]] = None,
) -> str:
    by_group: Dict[str, List[PersonLinkageDecisionV1]] = {g: [] for g, _, _ in GROUPS}
    for d in decisions:
        by_group[d.group].append(d)
    sections = []
    for g, heading, blurb in GROUPS:
        items = sorted(by_group[g], key=lambda d: (d.anchor_label, -(d.score or 0.0), d.candidate_label, d.decision_id))
        default = "approve" if g != "needs_your_call" else ""
        cards = "".join(_card(d, default) for d in items)
        body = cards or "<p class=muted>Nothing in this group.</p>"
        if g == "auto_reject" and items:
            body = f"<details><summary>Show {len(items)} cards</summary>{cards}</details>"
        sections.append(f"<section id={g}><h2>{_e(heading)} ({len(items)})</h2><p>{_e(blurb)}</p>{body}</section>")
    info = ""
    if info_rows:
        trs = "".join(
            f"<tr><td>{_e(r.get('name'))}</td><td>{_e(r.get('mentions'))}</td><td>{_e(r.get('first_seen'))}</td><td>{_e(r.get('example'))}</td></tr>"
            for r in info_rows
        )
        info = (
            "<section id=new_people><h2>People in your words with no node yet (no action now)</h2>"
            "<p>Names the extractor found in your chat messages that have no person node in the graph. "
            "Promotion and study come in a later phase; this is here so you can see who Orion would start with.</p>"
            f"<table class=info><tr><th>name</th><th>mentions</th><th>first seen</th><th>example</th></tr>{trs}</table></section>"
        )
    summary = "".join(f"<p>{_e(s)}</p>" for s in summary_lines)
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head>
<body data-run="{_e(run_id)}">
<header><h1>{_e(title)}</h1>{summary}<p id=progress></p><button id=export type=button>Export decisions (CSV)</button></header>
<main>{''.join(sections)}{info}</main>
<script>{_JS}</script>
<script type="application/json" id=decisions-json>{_e(json.dumps([d.decision_id for d in decisions]))}</script>
</body></html>
"""
