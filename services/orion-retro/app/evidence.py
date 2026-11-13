"""Bounded append-only evidence; fail closed on quota or I/O failure."""
import json
from datetime import datetime, timezone


class Evidence:
    def __init__(self, root, session, session_limit, total_limit):
        root.mkdir(parents=True, exist_ok=True)
        self.total = sum(p.stat().st_size for p in root.glob('*') if p.is_file())
        self.used = 0
        self.session_limit, self.total_limit = session_limit, total_limit
        self.path = root / f'{session}.jsonl'
        self.file = self.path.open('x', buffering=1)

    def write(self, event):
        row = dict(event, utc=datetime.now(timezone.utc).isoformat())
        data = json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n'
        size = len(data.encode())
        if self.used+size > self.session_limit or self.total+size > self.total_limit:
            raise OSError('evidence_quota')
        self.file.write(data)
        self.used += size; self.total += size

    def close(self):
        self.file.close()
