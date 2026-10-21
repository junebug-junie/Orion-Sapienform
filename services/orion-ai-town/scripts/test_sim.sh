#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$(mktemp -d)"
trap 'rm -r "$BUILD"' EXIT
TSC="${TSC:-$ROOT/upstream/node_modules/.bin/tsc}"
"$TSC" "$ROOT/upstream/convex/sim/runtime.ts" "$ROOT/upstream/convex/_generated/server.js" --outDir "$BUILD" --target es2020 --module commonjs --strict --skipLibCheck --allowJs --types node --typeRoots "$ROOT/upstream/node_modules/@types" --esModuleInterop --rootDir "$ROOT/upstream"
NODE_PATH="$ROOT/upstream/node_modules" TOWN_SIM_CORE="$BUILD/convex/sim/core.js" TOWN_SIM_RUNTIME="$BUILD/convex/sim/runtime.js" node --test "$ROOT/tests/sim_core.test.cjs" "$ROOT/tests/sim_runtime.test.cjs"
