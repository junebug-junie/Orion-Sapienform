#!/bin/sh
# Install a repo-aware front door; other repos still use the original CLI.
set -eu
GRAPHIFY_BIN=${GRAPHIFY_BIN:-$(command -v graphify)}
NATIVE="${GRAPHIFY_BIN}-orion-upstream"
if ! grep -q "# Orion Graphify local-storage launcher" "$GRAPHIFY_BIN"; then
    cp -p "$GRAPHIFY_BIN" "$NATIVE"
fi
PYTHON=$(head -n 1 "$NATIVE" | sed 's/^#!//')
[ -x "$PYTHON" ] || { echo "Expected Graphify's Python entry point at $NATIVE" >&2; exit 1; }
# Python writes quoted shell literals; paths containing whitespace stay intact.
python3 - "$GRAPHIFY_BIN" "$NATIVE" "$PYTHON" <<'PY'
import pathlib, shlex, sys
binary, native, interpreter = sys.argv[1:]
path = pathlib.Path(binary)
script = '''#!/bin/sh
# Orion Graphify local-storage launcher (reinstall after upgrading graphify).
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$ROOT" ] && [ -f "$ROOT/scripts/graphify_local.py" ]; then
    exec INTERPRETER "$ROOT/scripts/graphify_local.py" "$@"
fi
exec NATIVE "$@"
'''.replace('INTERPRETER', shlex.quote(interpreter)).replace('NATIVE', shlex.quote(native))
path.write_text(script)
path.chmod(0o755)
PY
python3 scripts/graphify_storage.py init
printf 'Installed local Graphify launcher: %s\n' "$GRAPHIFY_BIN"
