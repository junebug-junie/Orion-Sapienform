#!/bin/sh
set -eu
exec python -m kev.serve --run "${KEV_MODEL:-jaredpalmer/kev-0.8b}" --port 8009
