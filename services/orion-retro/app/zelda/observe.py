"""Read-only, fingerprint-gated Legend of Zelda (NES) observations: `zelda-1-ram-v1`.

Every address comes from aldonunez/zelda1-disassembly (src/Variables.inc, Z_05/Z_07.asm)
and was settled by a live probe with real controller input only (see VERDICTS and
docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md). The reader calls
retro_get_memory_data and never writes. A field that has not been settled live is
reported as 'unknown', never as 0. No reward, progress or success is inferred here.
"""
import ctypes as C
import hashlib

PROFILE = 'zelda-1-ram-v1'
# Operator ROMs: PRG0 and PRG1. Both produced byte-identical RAM timelines.
ROMS = {
    '8f72dc2e98572eb4ba7c3a902bca5f69c448fc4391837e5f8f0d4556280440ac': 'prg0',
    '89232edf4f9b52e3cb872094bc78973de080befca2ddea893b6e936066514d4e': 'prg1',
}
# Same pinned FCEUmm ac0d00a7 builds as the Nova reader (host build, container build).
CORES = {
    'ff0aa3f2e1550e7e5231fd0b175e2ad4e09046d5eab658e716cd81e79206fab2',
    'b8ba56fbeba5bf7658da89fba97ae1049eafe583320db490c2bf891934bdda22',
}
SYSTEM_RAM, SAVE_RAM = 2, 0
RAM_SIZE, SAVE_SIZE = 2048, 8192
COLS, ROWS = 32, 22
TILES = 0x530  # PlayAreaTiles ($6530): column-major, 22 rows per column
HUD = 0x40

MODES = {5: 'play', 7: 'scroll', 0x0B: 'cave', 0x10: 'stairs', 0x11: 'death',
         4: 'transition', 6: 'transition', 2: 'transition', 3: 'transition',
         9: 'transition', 0x0A: 'transition', 0x0C: 'transition',
         1: 'menu', 0x0E: 'menu', 8: 'menu', 0x0D: 'menu', 0: 'other'}
# Modes with a readable room tile map.
MAP_MODES = ('play', 'cave')
FACING = {1: 'E', 2: 'W', 4: 'S', 8: 'N'}

# Live verdict per exposed field. Only CONFIRMED fields are decoded; everything
# else stays 'unknown'. Update this table only with a live probe artifact.
VERDICTS = {
    'mode': 'CONFIRMED', 'level_overworld': 'CONFIRMED', 'level_dungeon': 'UNSETTLED',
    'room': 'CONFIRMED', 'next_room': 'CONFIRMED', 'link': 'CONFIRMED',
    'hearts': 'CONFIRMED', 'invincible_frames': 'CONFIRMED', 'sword': 'CONFIRMED',
    'enemies': 'CONFIRMED', 'walls': 'CONFIRMED',
    # 2026-10-22 live: a type-$60 drop picked up in room $78 moved $66D 0 -> 1 and
    # stayed; SELECT in play moved $E0 0 -> 1 -> 0. No bomb/key drop was seen in a
    # 15-room farming probe, and no dungeon was entered, so those stay unknown.
    'rupees': 'CONFIRMED', 'paused': 'CONFIRMED', 'drops': 'CONFIRMED',
    'keys': 'UNSETTLED', 'bombs': 'UNSETTLED', 'candle': 'UNSETTLED', 'raft': 'UNSETTLED',
    # Continue/Save/Retry screen (mode 8): the selection is GameSubmode $13 & 3
    # (0 Continue, 1 Save, 2 Retry), $80 = START accepted (Z_05 UpdateMode8ContinueQuestion_Full).
    'continue_menu': 'CONFIRMED',
}
CONTINUE_CHOICES = ('continue', 'save', 'retry')
ITEM_ADDR = {'sword': 0x657, 'bombs': 0x658, 'candle': 0x65B, 'raft': 0x660,
             'rupees': 0x66D, 'keys': 0x66E}
ENEMY_SLOTS = range(1, 12)  # object slots 1..11; slot 0 is Link


def tile_of(x, y):
    """Shared tile convention: the game's collision hotspot (GetCollidableTile,
    Y + $B), rounded to the nearest 8 px cell. Used by the motor, atlas and planner map."""
    return (x + 4) // 8, (y + 11 - HUD + 4) // 8


def supported(identity):
    return identity.get('rom_sha256') in ROMS and identity.get('core_sha256') in CORES


def _settled(name):
    return VERDICTS.get(name) == 'CONFIRMED'


class ZeldaObserver:
    """Reads one RAM + save-RAM snapshot per call. Construct only inside the worker."""

    def __init__(self, core):
        self.core = core
        self.supported = supported(core.identity)
        if self.supported:
            core.lib.retro_get_memory_data.argtypes = [C.c_uint]
            core.lib.retro_get_memory_data.restype = C.c_void_p
            core.lib.retro_get_memory_size.argtypes = [C.c_uint]
            core.lib.retro_get_memory_size.restype = C.c_size_t

    def _region(self, region, size):
        if self.core.lib.retro_get_memory_size(region) != size: return None
        pointer = self.core.lib.retro_get_memory_data(region)
        return C.string_at(pointer, size) if pointer else None

    def read(self, frame=None):
        if not self.supported:
            return {'status': 'unknown', 'reason': 'unsupported_rom_or_core'}
        ram, save = self._region(SYSTEM_RAM, RAM_SIZE), self._region(SAVE_RAM, SAVE_SIZE)
        if ram is None or save is None:
            return {'status': 'unknown', 'reason': 'ram_unavailable'}
        return decode(ram, save, frame)


def hearts(ram):
    values, partial = ram[0x66F], ram[0x670]
    full = values & 0x0F
    return dict(current=full + (1 if partial >= 0x80 else .5 if partial else 0),
                containers=(values >> 4) + 1)


def walls(ram, save):
    """22x32 grid, True = the game's own collision says Link cannot walk there."""
    first_blocked = ram[0x34A]
    return [[save[TILES + c * ROWS + r] >= first_blocked for c in range(COLS)]
            for r in range(ROWS)]


def walls_hash(grid):
    bits = b''.join(int(''.join('1' if b else '0' for b in row), 2).to_bytes(4, 'big') for row in grid)
    return hashlib.sha256(bits).hexdigest()[:16]


def enemies(ram):
    rows = []
    for i in ENEMY_SLOTS:
        kind = ram[0x34F + i]
        # Live probe: a killed enemy reads type 0 or a death/drop object (>= $40).
        if 0 < kind < 0x40 and ram[0x485 + i]:
            x, y = ram[0x70 + i], ram[0x84 + i]
            rows.append(dict(slot=i, type=kind, x=x, y=y, tile=list(tile_of(x, y)), hp=ram[0x485 + i]))
    return rows


def drops(ram):
    """Item-drop objects (type $60). Walking onto one collects it; contents unknown."""
    rows = []
    for i in ENEMY_SLOTS:
        if ram[0x34F + i] == 0x60:
            x, y = ram[0x70 + i], ram[0x84 + i]
            rows.append(dict(slot=i, x=x, y=y, tile=list(tile_of(x, y))))
    return rows


def decode(ram, save, frame=None, include_walls=True):
    if len(ram) != RAM_SIZE or len(save) != SAVE_SIZE:
        return {'status': 'unknown', 'reason': 'ram_unavailable'}
    code = ram[0x12]
    mode = MODES.get(code, 'other')
    # Live: cave mode $B runs load submodes 2..8 (Link frozen at his overworld spot)
    # before submode 0 hands control over. Only submode 0 is playable.
    if mode in MAP_MODES and ram[0x13] != 0: mode = 'transition'
    if ram[0x66F] >> 4 > 15 or ram[0x34A] == 0:
        return {'status': 'unknown', 'reason': 'unqualified_game_state'}
    level = ram[0x10]
    level_value = (0 if level == 0 else
                   level if 1 <= level <= 9 and _settled('level_dungeon') else 'unknown')
    items = {name: (ram[addr] if _settled(name) else 'unknown') for name, addr in ITEM_ADDR.items()}
    # Live 2026-10-22: Link ignores input while the cave person's text types out
    # (ObjState+1 $AD = 0 price transfer, 1 textbox, 3 item cue, 7 textbox; Link moved
    # 1 px in 15 frames of state 0) and while an item is held up
    # (ItemLiftTimer $506 counts 127 -> 0). Overworld $AD is an enemy's state, unused here.
    frozen = (ram[0x506] != 0 or (mode == 'cave' and ram[0xAD] in (0, 1, 3, 7))
              or (_settled('paused') and ram[0xE0] != 0))
    out = dict(status='observed', profile=PROFILE, frame=frame, mode=mode, mode_code=code,
               submode=ram[0x13],
               level=level_value, room=ram[0xEB], next_room=ram[0xEC],
               paused=bool(ram[0xE0]) if _settled('paused') else 'unknown',
               link=dict(x=ram[0x70], y=ram[0x84], tile=list(tile_of(ram[0x70], ram[0x84])),
                         facing=FACING.get(ram[0x98], 'unknown'), invincible_frames=ram[0x4F0],
                         controllable=mode in MAP_MODES and not frozen),
               hearts=hearts(ram), items=items, enemies=enemies(ram),
               drops=drops(ram) if mode in MAP_MODES else [],
               menu=(dict(kind='continue', cursor=CONTINUE_CHOICES[ram[0x13] & 3] if ram[0x13] & 3 < 3 else 'unknown',
                          activated=bool(ram[0x13] & 0x80))
                     if code == 8 and _settled('continue_menu') else None))
    if mode in MAP_MODES:
        grid = walls(ram, save)
        out['walls_hash'] = walls_hash(grid)
        if include_walls: out['walls'] = grid
    else:
        out['walls_hash'] = None
    return out


def summary(observation):
    """Status/planner view: everything except the wall grid."""
    return {k: v for k, v in observation.items() if k != 'walls'}


def digest(observation):
    """Stable short digest of an observation, for goal_finished start/end records."""
    if observation.get('status') != 'observed': return 'unknown'
    keys = ('mode_code', 'level', 'room', 'link', 'hearts', 'items', 'enemies', 'walls_hash')
    text = repr([observation.get(k) for k in keys]).encode()
    return hashlib.sha256(text).hexdigest()[:16]
