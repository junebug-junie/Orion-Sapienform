# PR: Hub Mind Tab, Modal Drill-Down, and Safe Recent-Run Analytics

**Branch:** `feat/hub-mind-tab-modal` → `main`

> **Status: stub** — fill in before opening the PR.

---

## Summary

- Delivers the Hub Mind experience end-to-end: a dedicated Mind tab surfacing session-scoped recent run analytics, a chat-anchored modal drill-down per turn, and browser-local default mind-on-send preferences.
- Fixes route ordering and missing session headers that caused the `recent` endpoint to be shadowed by the `/{mind_run_id}` path-param route and queries to create unrelated sessions.

---

## Changes

### `services/orion-hub`

**`scripts/mind_routes.py`**
- _[ TODO: describe new/changed routes — session-scoped recent runs endpoint, mind run lookup, etc. ]_
- `GET /api/mind/runs/recent` moved before `GET /api/mind/runs/{mind_run_id}` to prevent FastAPI shadowing.
- All three Mind fetch calls now forward `X-Orion-Session-Id` so queries bind to the operator's existing session.

**`static/js/app.js`**
- _[ TODO: describe Mind tab wiring, modal open/close, default preferences, chat anchor logic ]_

**`templates/index.html`**
- _[ TODO: describe new Mind tab scaffold, modal markup, accessibility attributes ]_

**`tests/test_mind_hub_tab.py`** _(new)_
- _[ TODO: describe tab-level smoke tests ]_

**`tests/test_mind_routes.py`** _(new)_
- _[ TODO: describe route-level unit tests — recent runs pagination, run lookup, session isolation ]_

**`docs/superpowers/specs/2026-05-03-hub-mind-tab-and-modal-design.md`**
- Spec updated to reflect delivered vs. deferred items.

---

## Test Plan

- [ ] `pytest services/orion-hub/tests/test_mind_routes.py` — all route tests pass
- [ ] `pytest services/orion-hub/tests/test_mind_hub_tab.py` — tab smoke tests pass
- [ ] Open Hub UI → Mind tab renders without errors
- [ ] Click a recent run row → modal opens with correct drill-down data for that turn
- [ ] Toggle mind-on-send default → preference persists across page reload (localStorage)
- [ ] `GET /api/mind/runs/recent` returns session-scoped runs (not cross-session bleed)
- [ ] `GET /api/mind/runs/{id}` returns 404 for unknown run ID, not the recent-list response
