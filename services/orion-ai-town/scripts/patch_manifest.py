#!/usr/bin/env python3
"""Recognize the whole applied stack; overlapping patches cannot reverse-check alone."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
up=root/'upstream'
marker=up/'.git/orion-patch-manifest.json'
inputs=sorted((root/'patches').glob('*.patch'))+sorted((root/'overlay').rglob('*.ts'))
inputs.append(root/'scripts/apply_upstream_patches.sh')
# Hash the actual rendered join patch, which depends on operator-local cards.
rendered=subprocess.check_output([sys.executable,str(root/'scripts/generate_descriptions.py'),'--render-human-patch'])
h=hashlib.sha256(rendered)
for p in inputs:h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
fingerprint=h.hexdigest()
if sys.argv[1]=='write':
    marker.write_text(json.dumps({'fingerprint':fingerprint}))
    print('written')
elif marker.exists():
    if json.loads(marker.read_text())['fingerprint']!=fingerprint:
        raise SystemExit('Patch bundle changed. Prepare a fresh pinned upstream clone; do not layer new patches over the old stack.')
    print('applied')
else:
    print('unrecorded')
