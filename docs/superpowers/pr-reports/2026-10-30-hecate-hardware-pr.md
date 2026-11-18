## Summary

- Fill in hecate's hardware now that it is online: product string NF5288M5-G3GPU01 (confirmed by Juniper, read from DMI), 2x Xeon Gold 6148 (2.4 GHz, 80 threads), 183 GB RAM (usable, `free -g`).
- Same facts in the README mesh table.

## Outcome moved

The body atlas "bodies" line for hecate now carries CPU and RAM, closing the open item left by PR #2985 (hecate was offline then).

## Files changed

- `config/biometrics/node_catalog.yaml`: hecate `hardware:` line.
- `README.md`: Hecate row.

## Tests run

```text
yaml.safe_load(node_catalog.yaml)   ok
pytest -q orion/body_atlas -k "catalog or host or source"   27 passed
```

Evidence: `ssh hecate@hecate` → lscpu "Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz", 2 sockets, 80 CPUs; `free -g` Mem 183; nvidia-smi "Tesla V100-SXM2-32GB"; `/sys/class/dmi/id/product_name` "NF5288M5-G3GPU01".

## Evals run

None; data-only change.

## Review findings fixed

None; two-line data change, review not material.

## Restart required

The atlas reads the catalog from the harness image; the post-pull rebuild covers it. If not: `scripts/safe_docker_build.sh orion-harness-governor up -d --build`

## Risks / concerns

- Severity: low. 183 GB is usable memory, not installed DIMM size (needs sudo dmidecode).
