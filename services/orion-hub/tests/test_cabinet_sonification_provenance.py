"""Cross-language source-selection parity against the real shared merger."""
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from orion.telemetry.cabinet_snapshot_merge import CabinetSnapshotSource, merge_cabinet_sensors_payload
from orion.telemetry.cabinet_sensors import extract_cabinet_measurements

HUB = Path(__file__).resolve().parents[1]


def test_recorded_browser_adapter_matches_shared_merger_and_extractor():
    rows = json.loads((HUB / 'evals/fixtures/cabinet-sonification-live.json').read_text())
    # Include partial stale source, newest competing channel and tied timestamps.
    for offset in (-6000, -1000, 0):
        row = json.loads(json.dumps(rows[0]))
        at = datetime.fromisoformat(row['snapshot']['received_at'].replace('Z', '+00:00')).timestamp()
        row['sources']['a']['snapshot']['received_at'] = datetime.fromtimestamp(at + offset / 1000, timezone.utc).isoformat()
        row['sources']['a']['snapshot']['frame']['magnetic'] = {'magnitude_ut': 1}
        row['sources']['b']['snapshot']['received_at'] = row['snapshot']['received_at']
        rows.append(row)
    script = """
const {frameFromPayload} = require(process.argv[1]);
let input = ''; process.stdin.on('data', d => input += d);
process.stdin.on('end', () => console.log(JSON.stringify(JSON.parse(input).map(p =>
  frameFromPayload(p, Math.max(...Object.values(p.sources).map(s => Date.parse(s.snapshot.received_at))) + 1000)))));
"""
    result = subprocess.run(['node', '-e', script, str(HUB / 'static/js/cabinet-sonification.js')],
                            input=json.dumps(rows), text=True, capture_output=True, check=True)
    frames = json.loads(result.stdout)
    for row, browser in zip(rows, frames):
        now = max(datetime.fromisoformat(s['snapshot']['received_at'].replace('Z', '+00:00')) for s in row['sources'].values()) + timedelta(seconds=1)
        sources = []
        for source_id, source in row['sources'].items():
            raw = source['snapshot']
            at = datetime.fromisoformat(raw['received_at'].replace('Z', '+00:00'))
            sources.append(CabinetSnapshotSource(source_id, raw['frame'], raw['received_at'],
                (now - at).total_seconds() > 5 or raw['status'] != 'ok', raw['status'], None, raw))
        merged = merge_cabinet_sensors_payload(sources)
        values = extract_cabinet_measurements(merged)
        for reading in browser['readings']:
            expected = values.get(reading['key'])
            if expected is None:
                assert reading['status'] != 'ok'
            else:
                assert reading['value'] == pytest.approx(expected)
                assert reading['status'] == 'ok'
