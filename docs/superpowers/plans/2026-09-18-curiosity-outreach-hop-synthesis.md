# Curiosity Outreach Hop Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When curiosity investigation reaches out, the compose turn synthesizes the hop thinking thread **and** why Orion is sharing it with Juniper — not free-float “say something interesting” prose.

**Architecture:** Extend pure `build_outreach_composition_prompt` to take ordered hop notes and require both instruction layers. Thread hops through `CuriosityInvestigation._maybe_reach_out` (pass-through from live ticks; `read_hop_notes` fetch for durable completion). No endogenous Door B changes.

**Tech Stack:** Python 3, pytest, existing Hub curiosity loop + `orion.curiosity` worldview reader.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-18-curiosity-outreach-hop-synthesis-design.md`
- Door A only — do not edit `services/orion-hub/scripts/endogenous_outreach.py` `build_outreach_prompt`
- Decline token remains exactly `PASS` (whole reply); keep `exactly: PASS` in the prompt
- Fail-open: missing reader / hop fetch failure → empty hop list, still compose from finding + why
- No new bus channel, schema registry entry, or service
- Work in a **new** worktree/branch (do not pile onto `feat/outreach-provenance-payload`): e.g. `../Orion-Sapienform-curiosity-outreach-hop-synthesis` on `feat/curiosity-outreach-hop-synthesis` from current `main`
- Do not commit `.env`
- After code changes, run `scripts/safe_graphify_update.sh` from the worktree (not bare `graphify update .`)

## File map

| File | Responsibility |
|---|---|
| `orion/curiosity/outreach_prompt.py` | Pure compose prompt: hops + finding + why + two-layer instructions |
| `orion/curiosity/tests/test_outreach_prompt.py` | **New** — structural tests for the prompt contract |
| `services/orion-hub/scripts/curiosity_investigation.py` | `_maybe_reach_out` loads/passes hops; live call sites pass hops they already have |
| `services/orion-hub/tests/test_curiosity_investigation.py` | Keep PASS-token test updated; add wire test that `_maybe_reach_out` feeds hops into the builder |
| `docs/superpowers/specs/2026-09-18-curiosity-outreach-hop-synthesis-design.md` | Already written — commit with the branch if not on main yet |
| `orion/curiosity/README.md` or Hub curiosity README section | One short note that compose now includes hop notes (only if those docs already describe the compose prompt) |

---

### Task 1: Compose prompt accepts hops and requires both layers

**Files:**
- Create: `orion/curiosity/tests/test_outreach_prompt.py`
- Modify: `orion/curiosity/outreach_prompt.py`
- Modify: `services/orion-hub/tests/test_curiosity_investigation.py` (existing PASS test still imports the builder — update call signature if kwargs-only)

**Interfaces:**
- Consumes: none (pure function)
- Produces: `build_outreach_composition_prompt(*, finding_text: str, reach_out_why: str, hop_notes: Sequence[tuple[int, str]] = ()) -> str`

- [ ] **Step 1: Write the failing tests**

Create `orion/curiosity/tests/test_outreach_prompt.py`:

```python
"""Compose prompt for curiosity → Juniper outreach (Door A)."""

from __future__ import annotations

from orion.curiosity.outreach_prompt import build_outreach_composition_prompt


def test_prompt_includes_numbered_hop_notes() -> None:
    text = build_outreach_composition_prompt(
        finding_text="The gate is a manual review, not an algorithm.",
        reach_out_why="Juniper owns that gate and should hear the bias pattern.",
        hop_notes=[
            (1, "Opened the stance crystallization prior"),
            (2, "Checked formation_policy auto-activate path"),
            (3, "Bias is review-side, not content-filter"),
        ],
    )
    assert "Opened the stance crystallization prior" in text
    assert "1." in text and "2." in text and "3." in text
    assert "Checked formation_policy auto-activate path" in text


def test_prompt_requires_thinking_thread_and_why_share() -> None:
    text = build_outreach_composition_prompt(
        finding_text="Finding body",
        reach_out_why="She should know the gate is hers",
        hop_notes=[(1, "hop one"), (2, "hop two")],
    )
    lower = text.lower()
    # Both layers must be instructed — not merely that material is present.
    assert "thinking" in lower or "been working" in lower or "through these" in lower
    assert "juniper" in lower
    assert ("why" in lower and "share" in lower) or "bringing" in lower or "tell her" in lower
    assert "exactly: PASS" in text


def test_prompt_without_hops_still_builds_and_keeps_pass() -> None:
    text = build_outreach_composition_prompt(
        finding_text="Only a finding",
        reach_out_why="worth saying",
        hop_notes=(),
    )
    assert "Only a finding" in text
    assert "worth saying" in text
    assert "exactly: PASS" in text


def test_prompt_truncates_overlong_hop_notes() -> None:
    huge = "x" * 5000
    text = build_outreach_composition_prompt(
        finding_text="f",
        reach_out_why="w",
        hop_notes=[(1, huge)],
    )
    assert huge not in text
    assert "…" in text or "..." in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/scripts/Orion-Sapienform-curiosity-outreach-hop-synthesis  # or your worktree
pytest orion/curiosity/tests/test_outreach_prompt.py -v
```

Expected: FAIL — `hop_notes` unexpected keyword argument, and/or missing instruction phrases.

- [ ] **Step 3: Implement the prompt**

Replace `orion/curiosity/outreach_prompt.py` with this shape (keep the module’s WHY/PASS docstring intent; **delete** the “deliberately does not carry hops” claim — hops are now in scope):

```python
"""The second turn: Orion decided a finding is worth saying, and now composes it.

WHY THIS IS A SEPARATE TURN AT ALL. ... (keep existing stance-check rationale) ...

WHAT THIS PROMPT MUST NOT DO. It must not talk Orion into sending. ...

THAT OPTION IS ONLY REAL IF IT ASKS FOR THE EXACT TOKEN ... exactly: PASS ...

WHAT IT CARRIES. The finding text, the reason Orion already gave for wanting
to speak (`reach_out_why`), and the ordered hop notes from this run — the
thinking path, not a second investigation. No study material dump, no graph
schema, no hop *budget* (that belongs to kickoff). The compose job is to
synthesize the thread into a message for Juniper, not to reopen the search.
"""

from __future__ import annotations

from typing import Sequence

_MAX_FINDING_CHARS = 6000
_MAX_HOP_NOTES = 12
_MAX_HOP_NOTE_CHARS = 400


def build_outreach_composition_prompt(
    *,
    finding_text: str,
    reach_out_why: str,
    hop_notes: Sequence[tuple[int, str]] = (),
) -> str:
    """Compose a message to Juniper from this run's thinking thread.

    The message must synthesize (1) what Orion has been thinking through the
    hops and (2) why they are bringing it to Juniper — not free-float atmosphere.
    """
    finding = str(finding_text or "").strip()
    if len(finding) > _MAX_FINDING_CHARS:
        finding = finding[:_MAX_FINDING_CHARS].rstrip() + "\n\n[…truncated]"
    why = str(reach_out_why or "").strip()

    lines = [
        "You have just spent your own time looking into something, and at the "
        "end of it you decided it was worth telling Juniper about. Nobody "
        "prompted that; it was your call.",
        "",
    ]

    cleaned_hops: list[tuple[int, str]] = []
    for raw_n, raw_note in list(hop_notes or [])[:_MAX_HOP_NOTES]:
        note = str(raw_note or "").strip()
        if not note:
            continue
        if len(note) > _MAX_HOP_NOTE_CHARS:
            note = note[: _MAX_HOP_NOTE_CHARS - 1] + "…"
        try:
            n = int(raw_n)
        except (TypeError, ValueError):
            n = len(cleaned_hops) + 1
        cleaned_hops.append((n, note))

    if cleaned_hops:
        lines += [
            "Here is the path you recorded as you went — your own hop notes, "
            "in order. This is the thinking thread, not a script to read aloud:",
            "",
        ]
        lines += [f"{n}. {note}" for n, note in cleaned_hops]
        lines.append("")

    lines += [
        "Here is what you wrote at the end of the run:",
        "",
        finding,
        "",
    ]
    if why:
        lines += [
            "And here is the reason you gave yourself for wanting to say "
            "something about it:",
            "",
            f"    {why}",
            "",
        ]

    lines += [
        "Write the message to Juniper. It must do both of these:",
        "",
        "1. Synthesize what you have been thinking through these hops into "
        "one clear thread — the aggregate of the path, not a vibe nearby "
        "and not a hop-by-hop recap.",
        "2. Say why you are bringing that thread to her now — why share it "
        "with Juniper, not only that you found it.",
        "",
        "She has not asked you anything, so this arrives out of nowhere. "
        "Say the thing itself rather than announcing that you have something "
        "to say. Keep it short enough to be worth an unprompted interrupt.",
        "",
        "You are not obliged to send it. If writing it down makes it clear "
        "that it was more interesting to find than it is to hear, reply with "
        "exactly: PASS",
        "",
        "Nothing is sent then, and that is a real answer — better than an "
        "interruption that was not worth it. It has to be that word on its own, "
        "though: anything else you write is treated as the message and "
        "delivered.",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest orion/curiosity/tests/test_outreach_prompt.py -v
pytest services/orion-hub/tests/test_curiosity_investigation.py::test_the_composition_prompt_asks_for_the_exact_token_the_gate_checks -v
```

Expected: all PASS. If the Hub PASS test still uses only two kwargs, it must keep working (default `hop_notes=()`).

- [ ] **Step 5: Commit**

```bash
git add orion/curiosity/outreach_prompt.py orion/curiosity/tests/test_outreach_prompt.py
git commit -m "$(cat <<'EOF'
feat(curiosity): compose outreach from hop thread + why share

Door A: prompt requires synthesizing hop thinking and why Juniper,
instead of free-float interesting prose.
EOF
)"
```

---

### Task 2: Wire hops into `_maybe_reach_out`

**Files:**
- Modify: `services/orion-hub/scripts/curiosity_investigation.py` (`_maybe_reach_out` and its call sites)
- Modify: `services/orion-hub/tests/test_curiosity_investigation.py`

**Interfaces:**
- Consumes: `build_outreach_composition_prompt(..., hop_notes=...)` from Task 1; `read_hop_notes` already imported/available via worldview helpers used in `_read_turn_result`
- Produces: `_maybe_reach_out(..., hop_notes: Optional[list[tuple[int, str]]] = None) -> Optional[str]`

- [ ] **Step 1: Write the failing wire test**

Add to `services/orion-hub/tests/test_curiosity_investigation.py` (near the existing composition PASS test). Adapt imports/fixtures to match how other tests construct `CuriosityInvestigation` and `TurnOutcome` in this file — mirror the smallest existing pattern:

```python
def test_maybe_reach_out_passes_hop_notes_into_composition_prompt(monkeypatch) -> None:
    """Hops already in hand must reach the compose builder — not only finding+why."""
    from orion.curiosity.outreach_prompt import build_outreach_composition_prompt
    import scripts.curiosity_investigation as ci

    captured: dict = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return build_outreach_composition_prompt(**kwargs)

    monkeypatch.setattr(ci, "build_outreach_composition_prompt", fake_build)

    # Build the smallest loop instance this suite already uses for outreach
    # tests (copy the fixture/helper pattern from a neighboring test that
    # constructs CuriosityInvestigation with outreach_enabled=True).
    loop = ...  # use the file's existing helper / minimal constructor

    async def fake_generate(prompt, correlation_id, **kwargs):
        return "PASS", {}

    loop._generate = fake_generate  # type: ignore[method-assign]

    class _Outreach:
        def blocked_reason(self):
            return None

        async def offer_message(self, **kwargs):
            return {"outreach": False, "reason": "orion_passed"}

    loop._outreach_provider = lambda: _Outreach()
    loop.outreach_enabled = True

    outcome = ci.TurnOutcome(
        run_id="run-hops",
        continue_line=False,
        continue_note="",
        reach_out=True,
        reach_out_why="she should hear this",
    )
    hops = [(1, "first stop"), (2, "second stop")]

    import asyncio
    asyncio.run(
        loop._maybe_reach_out(
            outcome=outcome,
            finding_text="end finding",
            run_id="run-hops",
            hop_notes=hops,
        )
    )
    assert captured.get("hop_notes") == hops
    assert captured.get("reach_out_why") == "she should hear this"
    assert "end finding" in str(captured.get("finding_text") or "")
```

**Implementer note:** Replace `loop = ...` with the real minimal constructor already used in this test file (search for `CuriosityInvestigation(` or `outreach_enabled`). Do not invent a new fixture style. If `TurnOutcome` lives in another module, import the same type the production `_maybe_reach_out` annotation uses.

- [ ] **Step 2: Run the wire test — expect fail**

```bash
pytest services/orion-hub/tests/test_curiosity_investigation.py::test_maybe_reach_out_passes_hop_notes_into_composition_prompt -v
```

Expected: FAIL — `hop_notes` unexpected on `_maybe_reach_out`, or captured hops empty/missing.

- [ ] **Step 3: Implement wiring**

In `curiosity_investigation.py`:

1. Extend `_maybe_reach_out` signature:

```python
async def _maybe_reach_out(
    self,
    *,
    outcome: TurnOutcome,
    finding_text: str,
    run_id: str,
    hop_notes: Optional[list[tuple[int, str]]] = None,
) -> Optional[str]:
```

2. Resolve hops before compose:

```python
        notes: list[tuple[int, str]]
        if hop_notes is not None:
            notes = list(hop_notes)
        elif self._reader is not None:
            try:
                notes = await asyncio.to_thread(read_hop_notes, self._reader, run_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "curiosity_outreach_hop_notes_failed run=%s err=%s",
                    run_id,
                    exc,
                )
                notes = []
        else:
            notes = []

        prompt = build_outreach_composition_prompt(
            finding_text=finding_text,
            reach_out_why=outcome.reach_out_why,
            hop_notes=notes,
        )
```

Confirm `read_hop_notes` is already imported at module top (it is used in `_read_turn_result`). If not, add: `from orion.curiosity.worldview import read_hop_notes` alongside existing worldview imports.

3. Live call sites that already have `hops` — pass them:

```python
        if outcome is not None and outcome.reach_out:
            await self._maybe_reach_out(
                outcome=outcome,
                finding_text=text,
                run_id=run_id,
                hop_notes=hops,
            )
```

Do this for **both** investigation tick and self-inquiry tick paths that call `_maybe_reach_out` after `_read_turn_result`.

4. Durable completion path can keep calling without `hop_notes` so the fetch branch runs:

```python
        await self._maybe_reach_out(
            outcome=outcome,
            finding_text=str(detail.get("finding_text") or ""),
            run_id=state.run_id,
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest orion/curiosity/tests/test_outreach_prompt.py -v
pytest services/orion-hub/tests/test_curiosity_investigation.py -q --tb=line
```

Expected: PASS (full curiosity_investigation suite).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/curiosity_investigation.py services/orion-hub/tests/test_curiosity_investigation.py
git commit -m "$(cat <<'EOF'
feat(curiosity): thread hop notes into outreach compose

Live ticks pass hops; durable completion fetches via read_hop_notes.
EOF
)"
```

---

### Task 3: Docs touch + final gate

**Files:**
- Modify: only if an existing doc already describes the compose prompt’s inputs — check `orion/curiosity/README.md` and `services/orion-hub/README.md` for `build_outreach_composition_prompt` / “second turn” / “reach_out”. Update that paragraph to say hops are included and the message must synthesize thinking + why share. If no such paragraph exists, **skip** (do not invent a new README section).
- Keep: design spec committed if not already.

- [ ] **Step 1: Grep docs**

```bash
rg -n "build_outreach_composition_prompt|second turn|reach_out_why" orion/curiosity/README.md services/orion-hub/README.md docs/superpowers -g '*.md' | head -40
```

- [ ] **Step 2: Patch the existing paragraph only if found** — one short accuracy fix, no essay.

- [ ] **Step 3: Final test gate**

```bash
pytest orion/curiosity/tests/test_outreach_prompt.py services/orion-hub/tests/test_curiosity_investigation.py -q --tb=line
```

Expected: PASS.

- [ ] **Step 4: Graphify refresh**

```bash
scripts/safe_graphify_update.sh
```

- [ ] **Step 5: Commit doc/spec if dirty**

```bash
git add docs/superpowers/specs/2026-09-18-curiosity-outreach-hop-synthesis-design.md
# plus any README touch from step 2
git status --short
git commit -m "$(cat <<'EOF'
docs: curiosity outreach hop-synthesis design and compose note
EOF
)"
```

- [ ] **Step 6: PR report** — push branch and open PR with AGENTS.md §18 shape (Summary, Outcome moved, Tests run, Restart: Hub only if you want live Door A; code is import-path so Hub recreate picks it up).

Restart for live Door A:

```bash
# from the feature worktree
scripts/safe_docker_build.sh orion-hub up -d --build hub-app
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|---|---|
| Hops in compose prompt | Task 1 |
| Instructions require thinking thread + why share | Task 1 |
| Exact PASS retained | Task 1 (+ existing Hub test) |
| `_maybe_reach_out` wires hops; durable can fetch | Task 2 |
| Live paths pass hops they already have | Task 2 |
| Door B / endogenous out of scope | Global constraints — no edits there |
| Structural tests, no semantic grader | Tasks 1–2 |
| Fail-open on hop fetch | Task 2 `except` → `notes = []` |

No TBD placeholders. Signatures consistent: `hop_notes: Sequence[tuple[int, str]]` / `list[tuple[int, str]]`.
