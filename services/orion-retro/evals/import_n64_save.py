"""Bring an operator's own N64 saves from another Mupen64Plus emulator into Retro.

Run inside the orion-retro image (it needs the N64 core), with the catalog and saves
mounted. Cartridge files (.eep/.mpk/.sra/.fla) become that game's battery.srm; a
save state (.st*, raw or gzip "M64+SAVE") becomes a server save, after it is restored
and seen to run. Refuses to replace an existing battery unless --replace.

  docker run --rm --network none --user "$(id -u):$(id -g)" -e PYTHONPATH=/app \\
    -v "$CATALOG":/catalog:ro -v "$SAVES":/saves -v "$SRC":/src:ro --tmpfs /tmp \\
    orion-retro-retro python /app/evals/import_n64_save.py --game-id ocarina-of-time --sra /src/oot.sra
"""
import argparse
import gzip
import json
from pathlib import Path

from orion.retro.probe import CoreProbe, digest
from app.saves import SaveStore
from settings import Settings

# Mupen64Plus-Next save_memory_data (libretro/libretro_memory.h at the pinned commit).
REGIONS = {'eep': (0, 0x800), 'mpk': (0x800, 0x20000), 'sra': (0x20800, 0x8000), 'fla': (0x28800, 0x20000)}


def battery_image(base, files):
    image = bytearray(base)
    if len(image) != 0x48800: raise ValueError(f'unexpected save memory size {len(image)}')
    for kind, data in files.items():
        offset, limit = REGIONS[kind]
        if not 0 < len(data) <= limit: raise ValueError(f'{kind} is {len(data)} bytes; at most {limit}')
        image[offset:offset+len(data)] = data
    return bytes(image)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--catalog', type=Path, default=Path('/catalog/catalog.json'))
    parser.add_argument('--saves', type=Path, default=Path('/saves'))
    parser.add_argument('--game-id', required=True)
    for kind in REGIONS: parser.add_argument(f'--{kind}', type=Path)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--replace', action='store_true', help='overwrite an existing battery.srm')
    args = parser.parse_args(argv)
    settings = Settings(saves=args.saves)
    row = next((r for r in json.loads(args.catalog.read_text()) if r['game_id'] == args.game_id), None)
    if row is None or row.get('system') != 'n64': raise SystemExit('game is not an N64 catalog entry')
    core_path = settings.core_for('n64')
    identity = dict(rom_sha256=row['rom_sha256'], core_sha256=digest(core_path))
    store = SaveStore(args.saves, settings.save_max_per_game, settings.save_limit('n64'))
    core = CoreProbe(core_path, args.catalog.parent/row['file'], **settings.core_host('n64'))
    try:
        files = {k: getattr(args, k).read_bytes() for k in REGIONS if getattr(args, k)}
        if files:
            if store.read_battery(args.game_id) is not None and not args.replace:
                raise SystemExit('battery.srm exists; pass --replace to overwrite it')
            store.write_battery(args.game_id, battery_image(core.save_ram(), files))
            print(json.dumps(dict(battery=args.game_id, regions=sorted(files))))
        if args.state:
            raw = args.state.read_bytes()
            state = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
            if state[:8] != b'M64+SAVE': raise SystemExit('not a Mupen64Plus save state')
            core.restore(state)
            frames = set()
            for _ in range(120): frames.add(core.step())
            if len(frames) < 2: raise SystemExit('restored state does not run (picture never changes)')
            core.restore(state)  # store the state as given, not 120 frames later
            meta = store.write(args.game_id, identity, core.snapshot(), frame=0, played_frames=0,
                               session_id='import:'+args.state.name[:40])
            print(json.dumps(dict(save_id=meta['save_id'], bytes=meta['bytes'], distinct_frames=len(frames))))
    finally:
        core.close()


if __name__ == '__main__':
    main()
