# Orion Platform Audit Scripts

These scripts generate evidence artifacts for platform drift and architectural review.

## Renamed from `scripts/platform/` (2026-08-12)

This package used to live at `scripts/platform/`. It was renamed because that
path **shadowed Python's stdlib `platform` module**: Python auto-inserts a
script's own directory at `sys.path[0]` for any `python3 scripts/<name>.py`
invocation, and at that priority `scripts/platform/` won over the real
stdlib module for anything importing `platform` from a script run that way
— including transitively (stdlib `uuid.py`'s own `import platform`, needed
by `redis`'s `asyncio` submodule). The failure mode wasn't a clean
`ImportError`; it was `AttributeError: module 'platform' has no attribute
'system'` on the first real stdlib attribute access, which is a confusing
thing to debug from the calling code with no obvious link back to this
directory.

By the time this was traced to its root cause, **~24 other files across the
repo already carried their own local workaround** for exactly this
collision (grep `shadows stdlib` or `sys.path.pop(0)` under `scripts/` and
`orion/` to find them) — none of them fixed the actual cause, they each
just individually deprioritized their own script's directory on `sys.path`.
Those workarounds are harmless to leave in place (they're now no-ops, not
bugs) and were deliberately **not** touched by the rename — removing ~24
files' worth of now-redundant guards is a separate, larger cleanup, not
bundled into this patch. See:

- `scripts/self_study_enrichment_hook.py`'s module docstring for the full
  incident writeup (the case that got this traced to its root cause).
- `scripts/check_scripts_dir_no_stdlib_shadow.py` — a new deterministic
  gate that fails if anything under `scripts/` ever collides with a stdlib
  module name again (this exact `platform` collision, or any future one).
- `docs/superpowers/pr-reports/2026-08-13-scripts-platform-stdlib-shadow-rename-pr.md`
  for the full PR history of this rename.

If you're reading this because something broke after this rename: check
whether the failure is an `AttributeError` on a stdlib-shaped attribute
access from a script run directly as `python3 scripts/<something>.py` — if
so, it's very unlikely to be caused by this rename (which *removed* a
collision, it didn't introduce one), and more likely a *pre-existing*,
still-undiscovered collision with some other name under `scripts/`. Run
`python3 scripts/check_scripts_dir_no_stdlib_shadow.py` to check.

## Output

All outputs are written under:

- `codex_reviews/<RUN_ID>/reports/`

## Usage

```bash
bash scripts/platform_audits/run_all_audits.sh audit_001
```

## Notes

- Channel extraction is **call-site based**: it only counts channels passed to publish/subscribe/psubscribe or YAML channel keys.
- It does **not** treat arbitrary `orion:` strings (RDF predicates, schema ids) as channels.
- Schema resolution is best-effort by class-name matching unless a schema registry exists.

## Recommended workflow

1) Run audits  
2) Fix drift  
3) Re-run audits until clean  
4) Run MTH scenarios per docs/platform_codex_testing.md
