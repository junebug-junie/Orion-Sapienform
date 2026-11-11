"""The market's consent gate (attention patch 3d): "is this act allowed, and who agreed?".

Spec: docs/superpowers/specs/2026-11-07-attention-patch3-market-decides-design.md, section 3 (approved
2026-11-10). Rules: config/physiology/consent.yaml (its header explains every field).

One pure, deterministic function, ``consent_verdict(want, request, rules, now) -> ConsentVerdictV1``,
answers ``allow``, ``refuse`` or ``needs_consent`` with the rule it used. Choice
(``orion.physiology.choice.decide``) calls it for a want that carries a ``consent_request``; a ``refuse``
or ``needs_consent`` makes the decision ``deferred``, reason ``consent``, and
``orion.physiology.binding.bind_verdict`` never admits that, not even at the window's end.

It takes over only the retired policy rung's (L8) consent job: the act's reversibility class and its
consent requirement. Heat, arousal, caps, quiet hours, the Juniper-first yield and the self-monitor act
gate stay where they are (decided item 6).

Branches, in order (the first that applies decides):

0. (in ``orion.physiology.choice.consent_gate``) a want WITHOUT a request whose (family, verb) has a rule ->
   refuse ``request_missing``: the guarantee does not rest on the producer remembering to ask;
1. no rules (consent.yaml missing or malformed) -> refuse ``rules_unavailable`` (fail closed);
2. the request names another want, family or verb than the want -> refuse ``request_mismatch``;
3. no rule for (family, verb): ``reversible`` -> allow ``no_rule_reversible``; any other class -> refuse
   ``no_rule_not_reversible`` (fail closed);
4. the request claims another reversibility class than its rule -> refuse ``reversibility_mismatch``;
5. the rule says ``refused`` (every ``irreversible`` rule does) -> refuse ``refused_by_rule``;
6. an attest the rule requires is missing -> refuse ``missing_attest:<names>``;
7. ``consent: none`` -> allow ``no_consent_needed`` (only a ``reversible`` rule may say none);
8. ``standing:<id>``: the standing consent exists and has not expired -> allow ``standing_consent``,
   ``consent_ref`` = ``standing:<id>``; missing -> needs_consent ``standing_consent_missing``; expired ->
   needs_consent ``standing_consent_expired``.

v1 has no per-act ask: ``needs_consent`` means "a standing consent is missing", and only a PR Juniper merges
adds one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from orion.schemas.physiology_want import REVERSIBILITY, ConsentRequestV1, ConsentVerdictV1, WantV1

SCHEMA_VERSION = 1
STANDING_PREFIX = "standing:"


@dataclass(frozen=True)
class StandingConsent:
    consent_id: str
    granted_by: str
    granted_on: str
    pr: str = ""
    note: str = ""
    expires: Optional[datetime] = None


@dataclass(frozen=True)
class ConsentRule:
    rule_id: str
    family: str
    verb: str
    reversibility: str
    consent: str                      # none | standing:<id> | refused
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsentRules:
    rules: Mapping[tuple[str, str], ConsentRule]
    standing: Mapping[str, StandingConsent] = field(default_factory=dict)

    def get(self, family: str, verb: str) -> Optional[ConsentRule]:
        return self.rules.get((family, verb))


def _expiry(raw: Any, where: str) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, date):
        # A date expires at the START of that day (UTC): "until 2026-12-01" means not on the 1st.
        return datetime.combine(raw, time.min, tzinfo=timezone.utc)
    if isinstance(raw, str):
        try:
            v = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{where}: expires must be an ISO date or datetime, got {raw!r}") from exc
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    raise ValueError(f"{where}: expires must be an ISO date or datetime, got {raw!r}")


def parse_consent(doc: Any) -> ConsentRules:
    """Strict: a rule that could fail open is a config error, not a default."""
    if not isinstance(doc, dict) or doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"consent config must be a mapping with version: {SCHEMA_VERSION}")
    raw_standing = doc.get("standing") or {}
    if not isinstance(raw_standing, dict):
        raise ValueError("standing: must be a mapping")
    standing: dict[str, StandingConsent] = {}
    for cid, body in raw_standing.items():
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"bad standing consent id {cid!r}")
        if not isinstance(body, dict):
            raise ValueError(f"standing.{cid}: must be a mapping")
        by, on = body.get("granted_by"), body.get("granted_on")
        if not by or not on:
            raise ValueError(f"standing.{cid}: granted_by and granted_on are required (who agreed, and when)")
        standing[cid] = StandingConsent(consent_id=cid, granted_by=str(by), granted_on=str(on),
                                        pr=str(body.get("pr") or ""), note=str(body.get("note") or ""),
                                        expires=_expiry(body.get("expires"), f"standing.{cid}"))
    raw_rules = doc.get("rules")
    if not isinstance(raw_rules, dict):
        raise ValueError("rules: must be a mapping")
    rules: dict[tuple[str, str], ConsentRule] = {}
    for rid, body in raw_rules.items():
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"bad rule id {rid!r}")
        if not isinstance(body, dict):
            raise ValueError(f"{rid}: must be a mapping")
        family, verb = body.get("family"), body.get("verb")
        if not isinstance(family, str) or not family or not isinstance(verb, str) or not verb:
            raise ValueError(f"{rid}: family and verb are required")
        rev = body.get("reversibility")
        if rev not in REVERSIBILITY:
            raise ValueError(f"{rid}: reversibility must be one of {list(REVERSIBILITY)}, got {rev!r}")
        consent = body.get("consent")
        if not isinstance(consent, str) or not (
                consent in ("none", "refused") or (consent.startswith(STANDING_PREFIX) and len(consent) > len(STANDING_PREFIX))):
            raise ValueError(f"{rid}: consent must be none, refused or standing:<id>, got {consent!r}")
        if rev == "irreversible" and consent != "refused":
            raise ValueError(f"{rid}: an irreversible act must be consent: refused (v1)")
        if consent == "none" and rev != "reversible":
            raise ValueError(f"{rid}: only a reversible act may need no consent (consent: none)")
        requires = body.get("requires") or []
        if not isinstance(requires, list) or not all(isinstance(x, str) and x for x in requires):
            raise ValueError(f"{rid}: requires must be a list of attest names")
        key = (family, verb)
        if key in rules:
            raise ValueError(f"{rid}: ({family}, {verb}) already has rule {rules[key].rule_id}")
        rules[key] = ConsentRule(rule_id=rid, family=family, verb=verb, reversibility=rev, consent=consent,
                                 requires=tuple(dict.fromkeys(requires)))
    used = {r.consent[len(STANDING_PREFIX):] for r in rules.values() if r.consent.startswith(STANDING_PREFIX)}
    unused = sorted(set(standing) - used)
    if unused:
        # A consent that no rule reads never takes effect: most likely a typo in Juniper's PR. Refuse the file,
        # so the PR's own CI fails instead of the consent silently reading as missing.
        raise ValueError(f"standing consents not used by any rule: {unused} (rules ask for {sorted(used)})")
    return ConsentRules(rules=rules, standing=standing)


def load_consent(path: str | Path) -> tuple[Optional[ConsentRules], Optional[str]]:
    """(rules, None) or (None, reason). Never raises."""
    import yaml

    p = Path(path)
    try:
        text = p.read_text()
    except FileNotFoundError:
        return None, f"missing:{p.name}"
    except OSError as exc:
        return None, f"read_failed:{p.name}:{type(exc).__name__}"
    try:
        return parse_consent(yaml.safe_load(text)), None
    except Exception as exc:  # noqa: BLE001 -- a malformed config is a reason, not a crash
        return None, f"malformed:{p.name}:{str(exc)[:160]}"


def _v(verdict: str, reason: str, rule: Optional[ConsentRule] = None, consent_ref: Optional[str] = None) -> ConsentVerdictV1:
    return ConsentVerdictV1(verdict=verdict, rule_id=rule.rule_id if rule is not None else None, reason=reason,
                            consent_ref=consent_ref)


def consent_verdict(want: WantV1, request: ConsentRequestV1, rules: Optional[ConsentRules],
                    now: datetime) -> ConsentVerdictV1:
    """Pure and deterministic. The branch order is the module docstring's."""
    if rules is None:
        return _v("refuse", "rules_unavailable: config/physiology/consent.yaml was not read")
    if (request.want_id, request.family, request.verb) != (want.want_id, want.family, want.verb):
        return _v("refuse", f"request_mismatch: the request is for {request.want_id}/{request.family}/{request.verb}, "
                            f"the want is {want.want_id}/{want.family}/{want.verb}")
    rule = rules.get(request.family, request.verb)
    if rule is None:
        if request.reversibility == "reversible":
            return _v("allow", "no_rule_reversible: no rule, and the act is reversible")
        return _v("refuse", f"no_rule_not_reversible: no rule for ({request.family}, {request.verb}) and the act is "
                            f"{request.reversibility}")
    if request.reversibility != rule.reversibility:
        return _v("refuse", f"reversibility_mismatch: the request says {request.reversibility}, the rule says "
                            f"{rule.reversibility}", rule)
    if rule.consent == "refused":
        return _v("refuse", f"refused_by_rule: {rule.reversibility} acts of {rule.family}/{rule.verb} are refused", rule)
    missing = [a for a in rule.requires if a not in set(request.attests)]
    if missing:
        return _v("refuse", f"missing_attest:{','.join(missing)}", rule)
    if rule.consent == "none":
        return _v("allow", "no_consent_needed: a reversible act", rule)
    cid = rule.consent[len(STANDING_PREFIX):]
    ref = f"{STANDING_PREFIX}{cid}"
    sc = rules.standing.get(cid)
    if sc is None:
        return _v("needs_consent", f"standing_consent_missing: {ref} is not in consent.yaml (Juniper adds it by a PR)",
                  rule, ref)
    if sc.expires is not None and now >= sc.expires:
        return _v("needs_consent", f"standing_consent_expired: {ref} expired {sc.expires.isoformat()}", rule, ref)
    return _v("allow", f"standing_consent: granted by {sc.granted_by} on {sc.granted_on}"
                       + (f" ({sc.pr})" if sc.pr else ""), rule, ref)


__all__ = ["ConsentRule", "ConsentRules", "STANDING_PREFIX", "StandingConsent", "consent_verdict", "load_consent",
           "parse_consent"]
