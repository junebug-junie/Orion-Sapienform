import json, os, time
from pathlib import Path
rows = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        status = dict(line.split(':', 1) for line in (p/'status').read_text().splitlines() if ':' in line)
        row = {'pid': int(p.name), **{k: status.get(k, '').strip() for k in
               ('Name','VmRSS','RssAnon','RssFile','RssShmem','VmSwap','Cpus_allowed_list','Mems_allowed_list')}}
        by_node = {}
        for line in (p/'numa_maps').read_text().splitlines():
            fields = dict(x.split('=', 1) for x in line.split() if '=' in x)
            page_kib = int(fields.get('kernelpagesize_kB', '4'))
            for key,value in fields.items():
                if key.startswith('N') and key[1:].isdigit():
                    by_node[key] = by_node.get(key, 0) + int(value) * page_kib
        row['mapped_resident_kib_by_node'] = by_node
        rows.append(row)
    except (FileNotFoundError, ProcessLookupError):
        continue
    except PermissionError as exc:
        rows.append({'pid': int(p.name), 'error': str(exc)})
print(json.dumps({'unix_s': time.time(), 'processes': rows,
                  'note': 'Per-process mapped pages; shared pages may appear in multiple processes. Do not sum as unique physical RAM.'}))
