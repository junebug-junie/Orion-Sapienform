"""Orion's body-economy report as plain first-person text (``skills.physiology.state.v1``).

One fetch (``GET /physiology/report`` on orion-durable-runs) and one renderer, shared by the two
callers so Orion reads the same words on every path:

* ``python -P -m orion.dig body-economy ...``: Bash inside a chat turn or a curiosity run
  (orion-harness-governor). This is how Orion reaches it live today.
* the cortex-exec skill adapter ``skills.physiology.state.v1`` (Skill Runner, scheduler, verb plans).

Stdlib only (urllib), so the dig kit needs no new dependency. Read-only: the renderer changes
nothing and decides nothing. It reports what the report says, including what is unknown: a
section with no data prints its ``missing`` reason, never a blank or a zero.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping, Optional, Sequence

REPORT_URL_ENV = "ORION_PHYSIOLOGY_REPORT_URL"
DEFAULT_REPORT_URL = "http://orion-athena-durable-runs:8121"
SECTIONS = (
    "limits", "wants", "rhythm", "money", "unit_costs", "waste", "queue_wait", "decisions",
    "claude", "vram", "value_per_cost", "services", "processes",
)
#: What "how is my body" reads when no section is named: the headline constraints, money, wants,
#: waste, and anything down. Small enough for a chat turn.
OVERVIEW = ("limits", "money", "wants", "waste", "services")
WINDOWS = ("24h", "7d", "30d")

BIND_WORDS = {
    "slack": "slack",
    "binding": "BINDING (my own use is pushing it)",
    "exogenous_bind": "BINDING, not from my load",
    "stuck_at_ceiling": "BINDING, price stuck at its ceiling",
    "no_reading": "unknown (no reading)",
}


class ReportUnavailable(RuntimeError):
    """The report could not be read. The message is the plain reason to show Orion."""


def normalize_sections(raw: Any) -> tuple[str, ...]:
    """``None`` / "" / "overview" -> OVERVIEW. Accepts a list or a comma/space separated string.
    Raises ValueError naming the unknown section and the known ones."""
    if raw is None:
        return OVERVIEW
    items = raw if isinstance(raw, (list, tuple)) else str(raw).replace(",", " ").split()
    names = [str(s).strip().lower().replace("-", "_") for s in items if str(s).strip()]
    if not names or names == ["overview"]:
        return OVERVIEW
    if "all" in names:
        return SECTIONS
    unknown = [n for n in names if n not in SECTIONS and n != "overview"]
    if unknown:
        raise ValueError(f"unknown section(s): {', '.join(unknown)}. Known: overview, all, {', '.join(SECTIONS)}")
    out: list[str] = []
    for n in names:
        for s in (OVERVIEW if n == "overview" else (n,)):
            if s not in out:
                out.append(s)
    return tuple(out)


def report_url(base: Optional[str] = None) -> str:
    return (base or os.environ.get(REPORT_URL_ENV) or DEFAULT_REPORT_URL).rstrip("/")


def fetch_report(*, base_url: Optional[str] = None, window: str = "24h", sections: Sequence[str] = OVERVIEW,
                 family: Optional[str] = None, outcome: Optional[str] = None, limit: Optional[int] = None,
                 timeout_sec: float = 10.0) -> dict:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {', '.join(WINDOWS)} (got {window!r})")
    params: dict[str, Any] = {"window": window, "sections": ",".join(sections)}
    if family:
        params["family"] = family
    if outcome:
        params["outcome"] = outcome
    if limit:
        params["limit"] = int(limit)
    url = f"{report_url(base_url)}/physiology/report?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as resp:  # noqa: S310 -- fixed internal URL
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail") or ""
        except Exception:  # noqa: BLE001
            pass
        raise ReportUnavailable(f"the body report answered HTTP {exc.code} {detail}".strip()) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ReportUnavailable(f"the body report at {report_url(base_url)} is unreachable ({exc})") from exc
    except ValueError as exc:
        raise ReportUnavailable(f"the body report answered something that is not JSON ({exc})") from exc


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------


def _n(v: Any, nd: int = 1) -> str:
    if v is None:
        return "?"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{x:.0f}"


def _usd(v: Any, nd: int = 2) -> str:
    if v is None:
        return "?"
    x = float(v)
    if 0 < abs(x) < 0.01:
        return f"${x:.6f}".rstrip("0")
    return f"${x:,.{nd}f}"


def _dur(seconds: Any) -> str:
    if seconds is None:
        return "?"
    s = float(seconds)
    if s < 120:
        return f"{s:.0f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _missing_for(report: Mapping, section: str) -> list[str]:
    return [str(m.get("reason")) for m in report.get("missing") or [] if m.get("section") == section]


def _amount(v: Any, unit: Optional[str]) -> str:
    if v is None:
        return "?"
    if unit == "usd":
        return _usd(v)
    if unit == "c":
        return f"{_n(v, 2)} °C"
    if unit == "slot_s":
        return f"{_n(v, 0)} lease-s"
    return f"{_n(v, 2)} {unit or ''}".strip()


def _limit_line(lim: Mapping) -> str:
    now = lim.get("now") or {}
    state = BIND_WORDS.get(str(now.get("bind_state")), str(now.get("bind_state") or "no price"))
    why = str(now.get("why") or "").strip()
    label = lim.get("label") or lim.get("constraint")
    if now.get("level") is None:
        return f"- {label}: {state}" + (f". {why}" if why else "")
    unit = lim.get("unit")
    level = f"{_amount(now.get('level'), unit)} against a line of {_amount(lim.get('line'), unit)}"
    return f"- {label}: {state}; {level}" + (f". {why}" if why else "")


def render_limits(report: Mapping, *, full: bool) -> list[str]:
    lims = report.get("limits") or []
    if not lims:
        return []
    out = ["Limits now:"]
    shown = 0
    hidden_lanes = []
    for lim in lims:
        state = (lim.get("now") or {}).get("bind_state")
        is_lane = str(lim.get("constraint", "")).startswith("lane:")
        if is_lane and not full and state == "slack":
            hidden_lanes.append(str(lim["constraint"]).split(":", 1)[1])
            continue
        out.append(_limit_line(lim))
        shown += 1
    if hidden_lanes:
        out.append(f"- GPU lanes with slack: {', '.join(hidden_lanes)}")
    return out


def render_money(report: Mapping) -> list[str]:
    m = report.get("money") or {}
    if m.get("mtd_usd") is None:
        return []
    line = (f"Money: {_usd(m.get('mtd_usd'))} spent on electricity this month against a {_usd(m.get('envelope_usd'), 0)} "
            f"envelope; forecast {_usd(m.get('forecast_usd'))} by month end. ")
    if m.get("floor_usd") is not None:
        line += f"What stays switched on alone reaches {_usd(m.get('floor_usd'))}. "
    if m.get("rate_usd_per_kwh") is not None:
        line += f"Electricity now costs {_usd(m.get('rate_usd_per_kwh'), 3)}/kWh."
    out = [line.strip()]
    days = m.get("daily_kwh") or []
    if days:
        tail = ", ".join(f"{d['d'][5:]} {_n(d['kwh'])}{' (part of the day)' if d.get('partial') else ''}"
                         for d in days[-7:])
        out.append(f"- kWh per day (local): {tail}")
    return out


def render_town(report: Mapping, *, full: bool) -> list[str]:
    """AI Town sessions (2026-10-30): today's town time against Orion's cap, and recent sessions."""
    town = (report.get("wants") or {}).get("town")
    if not isinstance(town, Mapping):
        return []
    cap = town.get("daily_cap_seconds")
    head = (f"AI Town today ({town.get('local_date', '?')}): I spent {_dur(town.get('orion_seconds_today'))}"
            + (f" of my {_dur(cap)} cap" if cap is not None else " (cap unknown)")
            + f"; Juniper opened it for {_dur(town.get('manual_seconds_today'))} (not counted).")
    if town.get("open_kind") == "orion":
        head += " A session I chose is open now."
    elif town.get("open_kind") == "manual":
        head += " Juniper has the town open now."
    out = [head]
    sessions = town.get("sessions") or []
    for row in sessions[: (20 if full else 4)]:
        who = "I chose it" if row.get("kind") == "orion" else "Juniper opened it"
        end = row.get("stop_reason") or "still open"
        out.append(f"- {str(row.get('started_at', '?'))[:16]}Z: {_dur(row.get('seconds'))}, {who}; "
                   f"ended: {end}; {row.get('utterances', 0)} line(s) I spoke")
    return out


def render_wants(report: Mapping, *, full: bool) -> list[str]:
    town = render_town(report, full=full)
    fams = (report.get("wants") or {}).get("families") or []
    if not fams:
        return town
    tot_f = sum(int(f.get("funded") or 0) for f in fams)
    tot_d = sum(int(f.get("deferred") or 0) for f in fams)
    out = [f"Wants (shadow bids; nothing obeys them yet): {tot_f} funded, {tot_d} deferred."]
    rows = fams if full else fams[:6]
    for f in rows:
        out.append(f"- {f['family']} (tier {f.get('tier', '?')}): {f.get('funded', 0)} funded, {f.get('deferred', 0)} deferred")
    if len(fams) > len(rows):
        out.append(f"- ... {len(fams) - len(rows)} more famil{'y' if len(fams) - len(rows) == 1 else 'ies'}")
    return out + town


def render_rhythm(report: Mapping) -> list[str]:
    cells = report.get("rhythm") or []
    if not cells:
        return []
    by_hour: dict[int, list[float]] = {}
    for c in cells:
        by_hour.setdefault(int(c["hour"]), [0.0, 0.0])
        by_hour[int(c["hour"])][0] += float(c.get("heavy") or 0.0)
        by_hour[int(c["hour"])][1] += float(c.get("total") or 0.0)
    heavy_total = sum(v[0] for v in by_hour.values())
    if heavy_total <= 0:
        return ["Rhythm: no deferrable (background) GPU work in the window."]
    top = sorted(by_hour.items(), key=lambda kv: -kv[1][0])[:4]
    hours = ", ".join(f"{h:02d}:00 ({kv[0] / heavy_total:.0%})" for h, kv in top)
    return [f"Rhythm: my deferrable GPU work ({_dur(heavy_total)} of leases) ran most at local {hours}."]


def render_unit_costs(report: Mapping) -> list[str]:
    rows = report.get("unit_costs") or []
    if not rows:
        return []
    out = ["Unit costs:"]
    for r in rows:
        bits = []
        if r.get("usd") is not None:
            bits.append(_usd(r["usd"], 4))
        if r.get("wh") is not None:
            bits.append(f"{_n(r['wh'], 2)} W·h")
        if r.get("heat_c") is not None:
            bits.append(f"+{_n(r['heat_c'], 2)} °C cabinet")
        val = ", ".join(bits) if bits and r.get("status") != "not_measured" else "not measured"
        out.append(f"- {r['unit']}: {val} [{r.get('status')}: {r.get('source')}]")
    return out


def render_waste(report: Mapping, *, full: bool) -> list[str]:
    w = report.get("waste") or {}
    by = w.get("failed_lease_s_by_lane") or {}
    if not by:
        return []
    total = sum(float(v) for v in by.values())
    lanes = ", ".join(f"{k} {_dur(v)}" for k, v in sorted(by.items(), key=lambda kv: -float(kv[1])))
    out = [f"Waste: {_dur(total)} of GPU lease time ended without success ({lanes})."]
    for t in (w.get("top") or [])[: (10 if full else 3)]:
        out.append(f"- {t.get('holder')} on {t.get('role')}: held {_dur(t.get('held_s'))}, ended {t.get('reason')} at {str(t.get('at'))[:16]}")
    return out


def render_queue_wait(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("queue_wait") or []
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: -(float(r.get("p90_s") or 0.0)))
    slow = [r for r in ranked if float(r.get("p90_s") or 0.0) >= 1.0]
    out = ["Queue wait for a GPU lease (median / 90th percentile):"]
    shown = slow[: (15 if full else 6)]
    for r in shown:
        who = f"{r['family']} (by holder)" if r.get("family_source") == "holder" else r["family"]
        out.append(f"- {who} on {r['lane']}: {_dur(r.get('p50_s'))} / {_dur(r.get('p90_s'))} (n={r.get('n')})")
    fast = [r for r in ranked if float(r.get("p90_s") or 0.0) < 1.0]
    if fast:
        out.append(f"- {len(fast)} more lane/holder pairs ({sum(int(r.get('n') or 0) for r in fast)} leases) had a "
                   "90th-percentile wait under 1 s")
    if len(slow) > len(shown):
        out.append(f"- {len(slow) - len(shown)} slower pairs not shown (see the full report)")
    return out


def render_decisions(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("decisions") or []
    if not rows:
        return []
    out = [f"Bid decisions, newest first ({len(rows)} shown):"]
    for d in rows[: (len(rows) if full else 5)]:
        why = str(d.get("why") or "")
        if len(why) > 220:
            why = why[:217] + "..."
        bind = f", bound by {d['binding_constraint']}" if d.get("binding_constraint") else ""
        out.append(f"- {str(d.get('decided_at'))[:16]} {d['family']} {d.get('outcome')} ({d.get('reason')}{bind}): {why}")
    return out


def render_claude(report: Mapping) -> list[str]:
    c = report.get("claude") or {}
    t = c.get("window_tokens") or {}
    known = all(t.get(k) is not None for k in ("input", "output", "cache_read", "cache_write"))
    if not known and not c.get("status"):
        return []
    out = []
    if known:
        usd = f", about {_usd(c['usd_est'])} at API prices" if c.get("usd_est") is not None else ""
        out.append(f"Claude: {_n(t.get('output'), 0)} output tokens, {_n(t.get('input'), 0)} input, "
                   f"{_n(t.get('cache_read'), 0)} cache reads, {_n(t.get('cache_write'), 0)} cache writes{usd}.")
    else:
        out.append("Claude: token counts unknown for this window.")
    if c.get("limit_events"):
        out.append("- limit events: " + ", ".join(f"{e['kind']} at {str(e['at'])[:16]}" for e in c["limit_events"]))
    if c.get("status"):
        out.append(f"- {c['status']}")
    return out


def render_vram(report: Mapping) -> list[str]:
    cards = [c for c in report.get("vram") or [] if c.get("used_mb") is not None]
    if not cards:
        return []
    out = ["GPU memory headroom:"]
    for c in cards:
        roles = f" ({', '.join(c['roles'])})" if c.get("roles") else ""
        watts = f", drawing {_n(c['watts'], 0)} W" if c.get("watts") is not None else ""
        out.append(f"- {c['host']} {c['card']}{roles}: {_n(c.get('headroom_mb'), 0)} MB free of "
                   f"{_n(c.get('total_mb'), 0)} MB{watts}")
    return out


def render_processes(report: Mapping, *, full: bool) -> list[str]:
    rows = report.get("processes") or []
    if not rows:
        return []
    out = ["CPU watts above idle by container, now (largest first):"]
    for r in rows[: (len(rows) if full else 6)]:
        out.append(f"- {r['host']} {r['name']}: {_n(r.get('watts'), 1)} W")
    return out


def render_value_per_cost(report: Mapping) -> list[str]:
    rows = report.get("value_per_cost") or []
    if not rows:
        return []
    out = ["Value per cost (graded evidence the market uses):"]
    for r in rows:
        cost = f", {_usd(r['usd_per_act'], 6)} per act" if r.get("usd_per_act") is not None else ", cost per act not measured"
        out.append(f"- {r['family']}: value {_n(r.get('value'), 2)} over {r.get('evidence_n')} acts ({r.get('value_metric')}){cost}")
    return out


def render_services(report: Mapping) -> list[str]:
    s = report.get("services") or {}
    down = s.get("down") or []
    if not down:
        return []
    out = [f"Services the mesh guardian reported stopped in the last 30 days ({len(down)}):"]
    for d in down:
        where = f" on {d['host']}" if d.get("host") else ""
        out.append(f"- {d['name']}{where}: {d.get('state') or '?'} since {str(d.get('since') or '?')[:16]}")
    if s.get("basis"):
        out.append(f"- ({s['basis']})")
    return out


_RENDER = {
    "limits": lambda r, full: render_limits(r, full=full),
    "money": lambda r, full: render_money(r),
    "wants": lambda r, full: render_wants(r, full=full),
    "rhythm": lambda r, full: render_rhythm(r),
    "unit_costs": lambda r, full: render_unit_costs(r),
    "waste": lambda r, full: render_waste(r, full=full),
    "queue_wait": lambda r, full: render_queue_wait(r, full=full),
    "decisions": lambda r, full: render_decisions(r, full=full),
    "claude": lambda r, full: render_claude(r),
    "vram": lambda r, full: render_vram(r),
    "value_per_cost": lambda r, full: render_value_per_cost(r),
    "services": lambda r, full: render_services(r),
    "processes": lambda r, full: render_processes(r, full=full),
}
assert set(_RENDER) == set(SECTIONS)


def headline(report: Mapping) -> Optional[str]:
    """One sentence naming what binds right now, in the report's own words. A limit with no price
    or no reading is unknown, never counted as slack."""
    lims = report.get("limits") or []
    if not lims:
        return None
    state = lambda lim: (lim.get("now") or {}).get("bind_state")  # noqa: E731
    name = lambda lim: str(lim.get("label") or lim.get("constraint"))  # noqa: E731
    binding = [name(l) for l in lims if state(l) in ("binding", "exogenous_bind", "stuck_at_ceiling")]
    unknown = [name(l) for l in lims if state(l) not in ("binding", "exogenous_bind", "stuck_at_ceiling", "slack")]
    parts = [f"Binding now: {', '.join(binding)}." if binding else
             ("Nothing binds right now." if not unknown else "Nothing with a price binds right now.")]
    if unknown:
        parts.append(f"Unknown (no price or no reading): {', '.join(unknown)}.")
    return " ".join(parts)


def render_report_text(report: Mapping, sections: Iterable[str] = OVERVIEW) -> str:
    """Plain text, first person, one paragraph per asked section. A section with nothing to show
    prints its ``missing`` reason. One or two named sections show every row; more show the top rows."""
    secs = tuple(sections)
    # Every row only when one or two sections are asked; the overview and "all" stay turn-sized.
    full = len(secs) <= 2
    win = report.get("window") or {}
    fam = (report.get("query") or {}).get("family")
    scope = f", family {fam} (narrows wants, decisions, queue wait, value per cost, per-act costs)" if fam else ""
    lines = [f"My body economy, {str(win.get('start', '?'))[:16]} to {str(win.get('end', '?'))[:16]} UTC{scope} "
             f"(read {str(report.get('generated_at', '?'))[:19]}):"]
    if "limits" in secs:
        h = headline(report)
        if h:
            lines.append(h)
    # More sections than the overview (e.g. "all"): one summary line each, so every section fits.
    brief = len(secs) > len(OVERVIEW)
    for s in secs:
        body = _RENDER[s](report, full)
        reasons = _missing_for(report, s)
        if body and brief:
            head = body[0] if len(body) == 1 or not body[0].endswith(":") else f"{body[0]} {body[1].lstrip('- ')}"
            more = len(body) - (1 if head == body[0] else 2)
            lines.append(head + (f" (+{more} more lines)" if more > 0 else "")
                         + (f" [{len(reasons)} note(s)]" if reasons else ""))
        elif body:
            lines.extend(body)
            for r in reasons:
                lines.append(f"- note ({s}): {r}")
        elif reasons:
            lines.append(f"{s}: no data. " + "; ".join(reasons))
        else:
            lines.append(f"{s}: no data, and the report gave no reason (treat as unknown).")
    if brief:
        lines.append("(One line per section. Ask for one or two sections by name for every row and note.)")
    failed = [m for m in report.get("missing") or [] if m.get("section") not in secs and m.get("section") not in ("*",)
              and str(m.get("reason", "")).startswith("read failed")]
    for m in failed:
        lines.append(f"- {m['section']}: {m['reason']}")
    return "\n".join(lines)


def cap_text(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    """Keep a reply turn-sized (the dig kit's own cap). A cut is always said."""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [f"(showing {max_lines - 1} of {len(text.splitlines())} lines; "
                                          "ask for fewer sections to see the rest)"]
    out = "\n".join(lines)
    if len(out) > max_chars:
        note = "(cut at the size limit; ask for fewer sections to see the rest)"
        out = out[: max_chars - len(note) - 1].rsplit("\n", 1)[0] + "\n" + note
    return out


__all__ = [
    "cap_text",
    "DEFAULT_REPORT_URL",
    "OVERVIEW",
    "REPORT_URL_ENV",
    "ReportUnavailable",
    "SECTIONS",
    "WINDOWS",
    "fetch_report",
    "normalize_sections",
    "render_report_text",
]
