# Orion Platform Audit Scripts

These scripts generate evidence artifacts for platform drift and architectural review.

## Output

All outputs are written under:

- `codex_reviews/<RUN_ID>/reports/`

## Usage

```bash
bash scripts/platform/run_all_audits.sh audit_001
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
