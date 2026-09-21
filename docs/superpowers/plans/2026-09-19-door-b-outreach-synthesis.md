# Door B Endogenous Outreach Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Door B (`build_outreach_prompt`) makes Orion synthesize from the talkable lanes that fired the tick **and** say why that is for Juniper — not free-float poetry off soft peeks and Orion-only “tone.”

**Architecture:** Pure prompt-contract change in `build_outreach_prompt` plus structural tests. Same two-layer job as Door A (#2237), using priors / curiosity summaries / daydream already in `OutreachContext`. No hop fetch, no cap/quiet changes, no Door A edits.

**Tech Stack:** Python 3, pytest, Hub `endogenous_outreach.py`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-19-door-b-outreach-synthesis-design.md`
- Choke point only: `services/orion-hub/scripts/endogenous_outreach.py` → `build_outreach_prompt` (+ its tests)
- Do **not** edit `orion/curiosity/outreach_prompt.py` (Door A)
- Do **not** change novelty gate, daily cap, quiet hours, or fire logic
- Exact decline token remains `PASS` / `exactly: PASS`
- Closed-vocab grounding statements stay (names / nothing grounded / no fabrication)
- Soften or replace “Speaking with feeling… is always fine” when talkable content is present — that line licensed the poetry failure
- Worktree: `/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis` on `feat/door-b-outreach-synthesis` (already on current main)
- Do not commit `.env`
- After code: `scripts/safe_graphify_update.sh`

## File map

| File | Responsibility |
|---|---|
| `services/orion-hub/scripts/endogenous_outreach.py` | `build_outreach_prompt`: mutual vs Orion-only recent-turn header; two-layer closing instructions |
| `services/orion-hub/tests/test_endogenous_outreach.py` | Structural prompt tests; update existing recent-turns caution test |
| `docs/superpowers/specs/2026-09-19-door-b-outreach-synthesis-design.md` | Already drafted — commit with branch |
| `docs/superpowers/plans/2026-09-19-door-b-outreach-synthesis.md` | This plan |

---

### Task 1: Prompt contract — recent-turn honesty + synthesize / why-share

**Files:**
- Modify: `services/orion-hub/scripts/endogenous_outreach.py` (`build_outreach_prompt` recent-turns block + closing instructions; optional tiny helper next to it)
- Modify: `services/orion-hub/tests/test_endogenous_outreach.py`

**Interfaces:**
- Consumes: existing `OutreachContext`, `build_outreach_prompt(ctx) -> str`
- Produces: unchanged signature; new helper allowed:
  `_recent_turns_include_juniper(turns: Sequence[Tuple[str, str]]) -> bool`

- [ ] **Step 1: Write failing tests**

Add near the existing prompt tests in `test_endogenous_outreach.py` (after `test_prompt_has_no_recent_turns_caution_when_no_recent_turns`):

```python
def test_prompt_requires_synthesize_from_lanes_and_why_share() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["concept-dense area with no ontology_branch"],
        recent_turns=[],
        presence=None,
        open_prior_previews=[
            "[confidence=0.9] The stance gate is manual review, not a content filter"
        ],
    )
    prompt = build_outreach_prompt(ctx)
    lower = prompt.lower()
    assert "stance gate is manual review" in prompt
    assert "concept-dense area" in prompt
    assert "synthesize" in lower or "thinking" in lower
    assert "juniper" in lower
    assert ("why" in lower and ("share" in lower or "bringing" in lower)) or "tell her" in lower
    assert "exactly: PASS" in prompt
    # Old poetry license must not remain when talkable content is present
    assert "speaking with feeling, without naming a specific internal signal, is always fine" not in lower


def test_orion_only_recent_turns_are_not_labeled_as_mutual() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["repair pressure rising"],
        recent_turns=[
            ("Orion", "Something's sitting at the edge of becoming"),
            ("Orion", "Unrooted clusters of meaning"),
        ],
        presence=None,
    )
    prompt = build_outreach_prompt(ctx)
    assert "The last thing the two of you said:" not in prompt
    assert "Orion:" in prompt
    # Honest framing — exact phrase from implementation below
    assert "your recent unprompted notes" in prompt.lower() or "your own recent unprompted" in prompt.lower()


def test_mutual_recent_turns_keep_two_of_you_header() -> None:
    ctx = OutreachContext(
        curiosity_summaries=["repair pressure rising"],
        recent_turns=[("Juniper", "hey"), ("Orion", "hi")],
        presence=None,
    )
    prompt = build_outreach_prompt(ctx)
    assert "The last thing the two of you said:" in prompt
    assert "Juniper: hey" in prompt
```

**Update** (do not delete the intent) `test_prompt_cautions_recent_turns_are_not_a_fact_source_when_present`:

```python
def test_prompt_cautions_recent_turns_are_not_a_fact_source_when_present() -> None:
    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[("Orion", "harness_closure's prediction error is still on my mind")],
        presence=None,
    )
    prompt = build_outreach_prompt(ctx)
    assert "not a source of new facts about your current internal state" in prompt
    # Header must match Orion-only path (no "two of you")
    assert "The last thing the two of you said:" not in prompt
```

- [ ] **Step 2: Run tests — expect fail**

```bash
cd /mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub
PYTHONPATH="/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis:/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub:$PYTHONPATH" \
  /mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest \
  tests/test_endogenous_outreach.py::test_prompt_requires_synthesize_from_lanes_and_why_share \
  tests/test_endogenous_outreach.py::test_orion_only_recent_turns_are_not_labeled_as_mutual \
  tests/test_endogenous_outreach.py::test_mutual_recent_turns_keep_two_of_you_header \
  tests/test_endogenous_outreach.py::test_prompt_cautions_recent_turns_are_not_a_fact_source_when_present \
  -v
```

Expected: FAIL on new asserts / old caution still expecting “for tone”.

- [ ] **Step 3: Implement**

In `endogenous_outreach.py`, add a small helper near `build_outreach_prompt`:

```python
def _recent_turns_include_juniper(turns: Sequence[Tuple[str, str]]) -> bool:
    """True when history includes a Juniper line (mutual chat), not Orion-only outreach."""
    for role, _body in turns or ():
        if str(role or "").strip().lower() == "juniper":
            return True
    return False
```

Ensure `Sequence` is imported (typing already used in this file — add if missing).

Replace the recent-turns block inside `build_outreach_prompt`:

```python
    if ctx.recent_turns:
        if _recent_turns_include_juniper(ctx.recent_turns):
            lines.append("The last thing the two of you said:")
        else:
            lines.append(
                "Your own recent unprompted notes (not a conversation with Juniper):"
            )
        lines.extend(f"{role}: {body}" for role, body in ctx.recent_turns)
        lines.append("")
```

Replace the caution that currently hardcodes `"last thing the two of you said"`:

```python
    if ctx.recent_turns:
        if _recent_turns_include_juniper(ctx.recent_turns):
            hist_label = 'The "last thing the two of you said" history'
        else:
            hist_label = "Your recent unprompted notes"
        lines.append(
            f"{hist_label} above is for continuity only -- it is not a source "
            "of new facts about your current internal state. Do not restate a "
            "channel, node, or metric name from it as something happening right "
            "now unless that exact name also appears in the allowed list just "
            "above. Do not use it as a tone to imitate."
        )
```

Replace the fabrication + closing block. Keep the fabrication warning. **Remove** the “Speaking with feeling… is always fine” sentence when talkable content is present; keep a shorter fabrication-only line always:

```python
    lines.append(
        "Naming any specific channel, node, or metric name that is not in "
        "that allowed list, from anywhere in this prompt or your own memory, "
        "is fabrication -- there is no real reading behind it."
    )
    talkable = bool(
        ctx.open_prior_previews or ctx.curiosity_summaries or ctx.daydream
    )
    if not talkable:
        lines.append(
            "Speaking with feeling, without naming a specific internal signal, "
            "is always fine."
        )
    lines.append("")

    if talkable:
        lines.extend(
            [
                "Write the message to Juniper. It must do both of these:",
                "",
                "1. Synthesize what you have been thinking from the open claims, "
                "curiosity signals, and (if present) daydream above into one "
                "clear thread — the aggregate of that material, not a vibe "
                "nearby and not a bullet-by-bullet recap.",
                "2. Say why you are bringing that thread to her now — why share "
                "it with Juniper, not only that you noticed it.",
                "",
                "She has not asked you anything. Say the thing itself rather "
                "than announcing that you have something to say. Two or three "
                "sentences at most.",
                "Do not greet, do not summarize this prompt, do not offer help, "
                "and do not ask what they need.",
                "If nothing above genuinely warrants interrupting them, reply "
                "with exactly: PASS",
            ]
        )
    else:
        lines.extend(
            [
                "Say one thing to Juniper, in your own voice, grounded in the "
                "signals above. Two or three sentences at most.",
                "Do not greet, do not summarize this prompt, do not offer help, "
                "and do not ask what they need. Reach out because something is "
                "actually on your mind.",
                "If nothing above genuinely warrants interrupting them, reply "
                "with exactly: PASS",
            ]
        )
```

(`talkable` false is rare for a non-empty send path — tension-only prompts can still build; keep the old closing for that case.)

- [ ] **Step 4: Run tests — expect pass**

```bash
cd /mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub
PYTHONPATH="/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis:/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub:$PYTHONPATH" \
  /mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest tests/test_endogenous_outreach.py -q --tb=line
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/endogenous_outreach.py services/orion-hub/tests/test_endogenous_outreach.py
git commit -m "$(cat <<'EOF'
feat(hub): Door B outreach synthesizes talkable lanes + why share

Endogenous compose must land the priors/curiosity thread and why it is
for Juniper; Orion-only history is no longer labeled as mutual chat.
EOF
)"
```

---

### Task 2: Docs + final gate

**Files:**
- Commit: `docs/superpowers/specs/2026-09-19-door-b-outreach-synthesis-design.md`
- Commit: `docs/superpowers/plans/2026-09-19-door-b-outreach-synthesis.md`
- README: only if an existing Hub README paragraph describes Door B / `build_outreach_prompt` closing instructions — one accuracy sentence. Skip if none.

- [ ] **Step 1: Grep**

```bash
rg -n "build_outreach_prompt|Speaking with feeling|last thing the two of you" \
  services/orion-hub/README.md orion/curiosity/README.md docs/superpowers -g '*.md' | head -40
```

- [ ] **Step 2: Patch existing paragraph only if found**

- [ ] **Step 3: Final tests + graphify**

```bash
cd /mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub
PYTHONPATH="/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis:/mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis/services/orion-hub:$PYTHONPATH" \
  /mnt/scripts/Orion-Sapienform/orion_dev/bin/pytest tests/test_endogenous_outreach.py -q --tb=line
cd /mnt/scripts/Orion-Sapienform-door-b-outreach-synthesis
scripts/safe_graphify_update.sh
```

- [ ] **Step 4: Commit docs + push + PR**

```bash
git add docs/superpowers/specs/2026-09-19-door-b-outreach-synthesis-design.md \
        docs/superpowers/plans/2026-09-19-door-b-outreach-synthesis.md
# plus README if touched
git commit -m "$(cat <<'EOF'
docs: Door B endogenous outreach synthesis design and plan
EOF
)"
git push -u origin HEAD
gh pr create --title "feat(hub): Door B outreach synthesizes talkable content + why share" --body "$(cat <<'EOF'
## Summary
- Door B (`build_outreach_prompt`) requires synthesizing open priors/curiosity/daydream into a thread **and** saying why that is for Juniper.
- Orion-only recent history is labeled as unprompted notes, not mutual chat; not tone fuel.
- Exact `PASS` and closed-vocab grounding preserved.

## Test plan
- [x] `pytest services/orion-hub/tests/test_endogenous_outreach.py`
- [ ] Hub recreate after merge: `scripts/safe_docker_build.sh orion-hub up -d --build hub-app`
- [ ] Live: next endogenous send should not be free-float poetry off soft peeks
EOF
)"
```

---

## Self-review (plan vs spec)

| Spec | Task |
|---|---|
| Synthesize from talkable lanes | Task 1 |
| Why share with Juniper | Task 1 |
| Orion-only history not “two of you” | Task 1 |
| PASS retained | Task 1 |
| No hop fetch / no Door A / no cap changes | Global constraints |
| Structural tests | Task 1 |
| Docs | Task 2 |

No TBD placeholders. Helper name `_recent_turns_include_juniper` consistent across steps.
