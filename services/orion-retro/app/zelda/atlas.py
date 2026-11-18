"""Orion's own atlas of Hyrule: `zelda-atlas-v1`, one JSON document per save lineage.

Written only from observed facts: a room change seen in RAM ($EB with $EC), an
observed item delta, an observed death or heart loss, and the navigator's bump
signal. It is never seeded and has no setter for arbitrary content. It stays out
of the memory store. File: <saves>/atlas/<lineage_id>.json, bounded at 1 MiB.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import uuid
from .observe import tile_of

PROFILE = 'zelda-atlas-v1'
MAX_BYTES = 1024 * 1024
LINEAGE = re.compile(r'^[0-9a-f]{32}$')
MAX_SESSIONS, MAX_ITEMS, MAX_BLOCKED, MAX_CAVES, MAX_TYPES = 200, 500, 64, 4, 32
FLUSH_SECONDS = 2.0
# Overworld and dungeon rooms are both laid out 16 columns wide.
DELTA = {-16: 'N', 16: 'S', -1: 'W', 1: 'E'}
OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
COUNTED = ('sword', 'rupees', 'keys', 'bombs', 'candle', 'raft')


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def room_key(level, room):
    return f'{level}:{room}'


def path_for(root, lineage_id):
    if not isinstance(lineage_id, str) or not LINEAGE.match(lineage_id):
        raise ValueError('invalid_lineage')
    return Path(root) / 'atlas' / f'{lineage_id}.json'


ROOM_KEYS = {'first_seen', 'first_session', 'visits', 'exits_seen', 'caves', 'learned_blocked',
             'enemy_types_seen', 'deaths', 'damage_taken', 'last_visit'}


def valid(doc):
    """Shape check, so a damaged file can never raise inside the worker frame loop."""
    try:
        return (doc.get('profile') == PROFILE and isinstance(doc.get('game'), dict)
                and LINEAGE.match(doc['lineage_id']) and isinstance(doc['items_gained'], list)
                and isinstance(doc['sessions'], list) and isinstance(doc['rooms'], dict)
                and all(isinstance(r, dict) and ROOM_KEYS <= set(r) and isinstance(r['exits_seen'], dict)
                        and set('NSEW') <= set(r['exits_seen']) and isinstance(r['caves'], list)
                        and isinstance(r['learned_blocked'], list) for r in doc['rooms'].values())
                and all(isinstance(s, dict) for s in doc['sessions']))
    except (AttributeError, KeyError, TypeError):
        return False


def load(root, lineage_id):
    try:
        path = path_for(root, lineage_id)
        if not path.is_file() or path.stat().st_size > MAX_BYTES: return None
        doc = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError): return None
    return doc if isinstance(doc, dict) and valid(doc) else None


def neighbourhood(doc, level, room):
    """Planner/Hub view: this room, its 4 neighbours, unvisited sides, caves, deaths.
    An unknown level or room is reported as unknown, never substituted."""
    rooms = {k: v for k, v in doc.get('rooms', {}).items() if not k.endswith(':cave')}
    if not isinstance(level, int) or not isinstance(room, int) or not 0 <= room <= 127:
        return dict(lineage_id=doc['lineage_id'], level='unknown' if not isinstance(level, int) else level,
                    room=room if isinstance(room, int) else None, current=None, adjacent={},
                    rooms_known=len(rooms), rooms_with_unvisited_sides=[], caves=[], deaths=[], map={})
    here = rooms.get(room_key(level, room))
    near = {}
    for delta, d in DELTA.items():
        other = room + delta
        if 0 <= other <= 127 and not (d in 'EW' and other // 16 != room // 16):  # 8x16 grid
            near[d] = dict(room=other, known=room_key(level, other) in rooms,
                           seen_exit=(here or {}).get('exits_seen', {}).get(d))
    # A side with no recorded crossing. It may be a wall; the atlas has no way to know.
    unexplored = sorted(k for k, v in rooms.items() if k.startswith(f'{level}:')
                        for d in 'NSEW' if v['exits_seen'].get(d) is None)
    return dict(lineage_id=doc['lineage_id'], level=level, room=room, current=here, adjacent=near,
                rooms_known=sum(1 for k in rooms if k.startswith(f'{level}:')),
                rooms_with_unvisited_sides=sorted(set(unexplored)),
                caves=[dict(room=k, **c) for k, v in rooms.items() if k.startswith(f'{level}:') for c in v['caves']],
                deaths=[dict(room=k, deaths=v['deaths']) for k, v in rooms.items()
                        if v['deaths'] and k.startswith(f'{level}:')],
                map={k: dict(visits=v['visits'], exits=v['exits_seen'],
                             caves=len(v['caves']), deaths=v['deaths'])
                     for k, v in rooms.items() if k.startswith(f'{level}:')})


class Atlas:
    def __init__(self, root, lineage_id, identity, session_id, now=time.monotonic, record=None):
        self.root = Path(root); self.identity = identity; self.session_id = session_id
        self.now = now; self.record = record or (lambda *a, **k: None)
        self.dirty = False; self.flushed_at = float('-inf'); self.disabled = None
        self.open(lineage_id)

    @staticmethod
    def new_lineage():
        return uuid.uuid4().hex

    def open(self, lineage_id):
        """Switch lineage (session start or a save load). Never mixes two lineages."""
        if getattr(self, 'doc', None): self.end_session(); self.flush(force=True)
        if not isinstance(lineage_id, str) or not LINEAGE.match(lineage_id):
            lineage_id = self.new_lineage()  # a damaged save field never reaches path_for
        doc = load(self.root, lineage_id)
        if doc and doc['game'] != {k: self.identity[k] for k in ('rom_sha256', 'core_sha256')}:
            # Never overwrite another build's atlas: start a new lineage instead.
            self.record('atlas_identity_mismatch', lineage_id=lineage_id)
            doc = None; lineage_id = self.new_lineage()
        self.lineage_id = lineage_id
        self.doc = doc or dict(profile=PROFILE, lineage_id=lineage_id,
                               game={k: self.identity[k] for k in ('rom_sha256', 'core_sha256')},
                               updated_at=utc(), rooms={}, items_gained=[], sessions=[])
        self.session = dict(session_id=self.session_id, start=utc(), end=None, rooms_new=0,
                            deaths=0, decisions=0, met=0, refuted=0,
                            frames_by_controller=dict(human=0, script=0, leisure=0, none=0))
        self.doc['sessions'].append(self.session)
        del self.doc['sessions'][:-MAX_SESSIONS]
        self.prev = None; self.place = None; self.scroll_to = None
        self.dirty = True

    # ---- observed-only writers ---------------------------------------------------
    def observe(self, obs, frame):
        if obs.get('status') != 'observed' or obs['level'] == 'unknown': return
        prev = self.prev; self.prev = obs
        mode = obs['mode']
        if mode == 'scroll': self.scroll_to = (obs['level'], obs['next_room'])
        if mode in ('play', 'cave'):
            place = (obs['level'], obs['room'], mode)
            if place != self.place:
                self._arrive(obs, frame)
                self.place = place
            if obs['link'].get('controllable'):
                self.unmark_standing(obs['level'], obs['room'], tile_of(obs['link']['x'], obs['link']['y']),
                                     mode, obs.get('walls_hash'))
        if prev is None or prev.get('status') != 'observed': return
        if mode == 'death' and prev['mode'] != 'death':
            self._room(prev)['deaths'] += 1; self.session['deaths'] += 1; self.dirty = True
        hp0, hp1 = prev['hearts']['current'], obs['hearts']['current']
        if hp1 < hp0 and prev['room'] == obs['room'] and prev['level'] == obs['level']:
            self._room(obs)['damage_taken'] += int((hp0 - hp1) * 2); self.dirty = True
        for item in COUNTED:
            a, b = prev['items'].get(item), obs['items'].get(item)
            if isinstance(a, int) and isinstance(b, int) and b > a:
                self._item(item, b - a, obs, frame)
        if prev['mode'] == 'play' and mode in ('stairs', 'cave') and prev['level'] == obs['level']:
            self._cave(prev)
        known = set(self._room(obs).get('enemy_types_seen', [])) if mode in ('play', 'cave') else set()
        for e in obs['enemies'] if mode in ('play', 'cave') else ():
            if e['type'] not in known and len(known) < MAX_TYPES:
                known.add(e['type']); self._room(obs)['enemy_types_seen'] = sorted(known); self.dirty = True

    def _room(self, obs, mode=None):
        key = room_key(obs['level'], obs['room'])
        if (mode or obs['mode']) == 'cave': key += ':cave'
        rooms = self.doc['rooms']
        if key not in rooms:
            rooms[key] = dict(first_seen=utc(), first_session=self.session_id, visits=0,
                              exits_seen={d: None for d in 'NSEW'}, caves=[], learned_blocked=[],
                              enemy_types_seen=[], deaths=0, damage_taken=0, last_visit=None)
            if not key.endswith(':cave'): self.session['rooms_new'] += 1
            self.dirty = True
        return rooms[key]

    def _arrive(self, obs, frame):
        room = self._room(obs)
        room['visits'] += 1; room['last_visit'] = utc(); self.dirty = True
        old = self.place
        scrolled = getattr(self, 'scroll_to', None) == (obs['level'], obs['room'])
        self.scroll_to = None
        # An exit is only a seen scroll ($EC) into this room, never a respawn or a load.
        if (scrolled and old and old[2] == 'play' and obs['mode'] == 'play' and old[0] == obs['level']
                and old[1] != obs['room'] and obs['room'] - old[1] in DELTA):
            d = DELTA[obs['room'] - old[1]]
            self.doc['rooms'][room_key(old[0], old[1])]['exits_seen'][d] = obs['room']
            room['exits_seen'][OPPOSITE[d]] = old[1]
        if old and old[2] == 'cave' and obs['mode'] == 'play':
            # Left a cave: mark the cave this room was entered through as entered.
            for c in room['caves']: c['entered'] = True

    def _cave(self, prev):
        room = self._room(prev); link = prev['link']
        c, r = tile_of(link['x'], link['y'])
        tile = [c, r - 1]  # the cave/stair tile Link walked up into
        if not any(c['tile'] == tile for c in room['caves']) and len(room['caves']) < MAX_CAVES:
            room['caves'].append(dict(tile=tile, entered=True, contents_seen=[]))
        self.cave_room = prev; self.dirty = True

    def _item(self, item, amount, obs, frame):
        row = dict(item=item, amount=amount, level=obs['level'], room=obs['room'], mode=obs['mode'],
                   frame=frame, session=self.session_id)
        self.doc['items_gained'].append(row)
        del self.doc['items_gained'][:-MAX_ITEMS]
        if obs['mode'] == 'cave' and getattr(self, 'cave_room', None):
            for c in self._room(self.cave_room, 'play')['caves']:
                if item not in c['contents_seen']: c['contents_seen'].append(item)
        self.dirty = True

    def mark_blocked(self, level, room, tile, mode='play', walls_hash=None):
        """A bump, keyed to the room layout it happened on: a changed layout (a bombed
        wall, an opened door) never inherits it."""
        if level == 'unknown': return
        key = room_key(level, room) + (':cave' if mode == 'cave' else '')
        r = self.doc['rooms'].get(key)
        if r is None: return
        entry = [tile[0], tile[1], walls_hash]
        if entry not in r['learned_blocked'] and len(r['learned_blocked']) < MAX_BLOCKED:
            r['learned_blocked'].append(entry); self.dirty = True

    def unmark_standing(self, level, room, tile, mode='play', walls_hash=None):
        """Link is standing on position (c, r) of this layout, so a learned block AT (c, r) was
        wrong (a bump against an enemy, a missed press) and heals. Learned blocks are Link
        positions (navigate._bump records the position he failed to reach; open_tile checks
        both wall columns itself), so standing here proves nothing about (c+1, r): healing it
        would erase every east-side bump on the next frame. Scans only this room's short list."""
        key = room_key(level, room) + (':cave' if mode == 'cave' else '')
        r = self.doc['rooms'].get(key)
        if not r or not r['learned_blocked']: return
        c, row = tile
        keep = [t for t in r['learned_blocked']
                if not (len(t) == 3 and t[2] == walls_hash and t[1] == row and t[0] == c)]
        if len(keep) != len(r['learned_blocked']):
            r['learned_blocked'] = keep; self.dirty = True

    def blocked(self, level, room, mode='play', walls_hash=None):
        key = room_key(level, room) + (':cave' if mode == 'cave' else '')
        return {(t[0], t[1]) for t in self.doc['rooms'].get(key, {}).get('learned_blocked', [])
                if len(t) == 3 and t[2] == walls_hash}

    def caves(self, level, room):
        return [tuple(c['tile']) for c in self.doc['rooms'].get(room_key(level, room), {}).get('caves', [])]

    def count_frame(self, controller):
        """One native frame played under this controller (human|script|leisure|none)."""
        counts = self.session['frames_by_controller']
        counts[controller if controller in counts else 'none'] += 1
        self.dirty = True

    def decision(self, verdict):
        self.session['decisions'] += 1
        if verdict in ('met', 'refuted'): self.session[verdict] += 1
        self.dirty = True

    # ---- persistence -------------------------------------------------------------
    def end_session(self):
        self.session['end'] = utc(); self.dirty = True

    def flush(self, force=False):
        if not self.dirty or self.disabled: return False
        if not force and self.now() - self.flushed_at < FLUSH_SECONDS: return False
        try:
            path = path_for(self.root, self.lineage_id)
            if not self.doc['rooms'] and not path.exists(): return False  # nothing observed yet
        except (OSError, ValueError) as exc:
            self.disabled = 'atlas_io_failed'; self.record('atlas_io_failed', error=type(exc).__name__)
            return False
        self.doc['updated_at'] = utc()
        data = json.dumps(self.doc, separators=(',', ':'))
        while len(data) > MAX_BYTES and len(self.doc['sessions']) > 1:
            # Drop the older half per pass: a bounded number of re-dumps in the frame loop.
            del self.doc['sessions'][:max(1, (len(self.doc['sessions']) - 1) // 2)]
            data = json.dumps(self.doc, separators=(',', ':'))
        if len(data) > MAX_BYTES:
            self.disabled = 'atlas_full'; self.record('atlas_full', lineage_id=self.lineage_id); return False
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            tmp.write_text(data); os.replace(tmp, path)
        except OSError as exc:
            # Atlas faults never kill the game; the session keeps playing without it.
            self.disabled = 'atlas_io_failed'; self.record('atlas_io_failed', error=type(exc).__name__)
            return False
        self.dirty = False; self.flushed_at = self.now()
        return True

    def public(self):
        return dict(lineage_id=self.lineage_id,
                    rooms_known=sum(1 for k in self.doc['rooms'] if not k.endswith(':cave')),
                    session=dict(self.session), disabled=self.disabled)
