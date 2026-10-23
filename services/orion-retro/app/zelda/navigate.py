"""Zelda goal motor (GoalV2): turns one accepted goal into frame-by-frame NES input.

Runs inside the native worker. Inputs are only the decoded observation and the atlas's
learned-blocked tiles. No GPU, no RAM writes, no reflex outside an accepted goal.

Grid convention (shared with the planner's text map): a "tile" is an 8 px cell of the
32x22 play area. Link's tile is his collision hotspot, the game's own GetCollidableTile:
column = X / 8, row = (Y + 11 - $40) / 8, rounded to the nearest cell. Link is 16 px
wide, so standing on tile (c, r) needs tiles (c, r) and (c+1, r) open.

Motion (live, 2026-10-22): Link moves 1-2 px per frame, and a perpendicular press first
slides him to the nearest 8 px grid line. So a waypoint counts as reached within
SLACK px, and a turn is simply the new direction; the game itself does the alignment.
"""
import heapq
from .observe import COLS, ROWS, HUD, summary, digest, tile_of

POLICY_ID = 'zelda-navigator-v1'
INTERVAL = 0  # send on every worker tick: one decision per native frame (60.0988 fps > 1/60 s)
GOALS = ('exit', 'enter', 'fight', 'retreat', 'collect', 'explore', 'wait')
PREDICTIONS = ('reach_room', 'enter_cave', 'item_gained', 'enemies_cleared', 'survive', 'unknown')
UP, DOWN, LEFT, RIGHT, A = 16, 32, 64, 128, 256
SELECT, START = 4, 8
DIRS = {'N': (0, -1, UP), 'S': (0, 1, DOWN), 'W': (-1, 0, LEFT), 'E': (1, 0, RIGHT)}
BUMP_FRAMES = 8          # spec: unchanged position for 8+ frames while moving = bump
STUCK_FRAMES = 120       # 2 s at 60 fps
OFFER_TTL = 30
SWING_COOLDOWN = 16
NEAR = 28                # px: sword reach without the beam
RETREAT_DISTANCE = 48
COLLECT_LINGER = 30
MAX_COL, MIN_ROW, MAX_ROW = COLS - 2, 1, ROWS - 1
SLACK = 3


def near(x, y, tx, ty):
    return abs(tx - x) <= SLACK and abs(ty - y) <= SLACK


def tile_xy(c, r):
    """Aligned pixel position for standing on tile (c, r): X multiple of 8, Y = 8k+5."""
    return c * 8, r * 8 - 11 + HUD


def open_tile(walls, blocked, c, r):
    return (0 <= c <= MAX_COL and MIN_ROW <= r <= MAX_ROW and not walls[r][c]
            and not walls[r][c + 1] and (c, r) not in blocked)


def neighbours(walls, blocked, c, r):
    for dc, dr, _ in DIRS.values():
        if open_tile(walls, blocked, c + dc, r + dr):
            yield c + dc, r + dr


def astar(walls, blocked, start, goals):
    """Shortest 4-connected path from start to any tile in goals. Returns tiles after start."""
    goals = set(goals)
    if not goals: return None
    if start in goals: return []
    def h(t): return min(abs(t[0] - g[0]) + abs(t[1] - g[1]) for g in goals)
    frontier = [(h(start), 0, start)]; came = {start: None}; cost = {start: 0}
    while frontier:
        _, g, current = heapq.heappop(frontier)
        if current in goals:
            path = []
            while current != start:
                path.append(current); current = came[current]
            return path[::-1]
        if g > cost[current]: continue
        for nxt in neighbours(walls, blocked, *current):
            if nxt not in cost or g + 1 < cost[nxt]:
                cost[nxt] = g + 1; came[nxt] = current
                heapq.heappush(frontier, (g + 1 + h(nxt), g + 1, nxt))
    return None


def reachable(walls, blocked, start, limit=10_000):
    seen = {start}; queue = [start]
    for t in queue:
        if len(seen) >= limit: break
        for n in neighbours(walls, blocked, *t):
            if n not in seen: seen.add(n); queue.append(n)
    return seen


def edge_tiles(walls, blocked, start, direction):
    """Reachable tiles furthest toward `direction`. For overworld exits this is the screen edge."""
    area = reachable(walls, blocked, start) | {start}
    axis, sign = (1, -1) if direction == 'N' else (1, 1) if direction == 'S' else \
                 (0, -1) if direction == 'W' else (0, 1)
    best = max(sign * t[axis] for t in area)
    return [t for t in area if sign * t[axis] == best], best


def on_screen_edge(tile, direction):
    c, r = tile
    return {'N': r == MIN_ROW, 'S': r == MAX_ROW, 'W': c == 0, 'E': c == MAX_COL}[direction]


def terrain(obs, learned=(), caves=()):
    """Planner map: 22 rows x 32 chars (~770 bytes). '#' blocked, '.' walkable, 'x' learned
    blocked by a bump on this layout, 'C' a cave this lineage has entered. Same tile grid as
    `tile`; row 0 is the top of the play area."""
    if 'walls' not in obs: return None
    rows = [['#' if b else '.' for b in row] for row in obs['walls']]
    for c, r in learned:
        if 0 <= r < ROWS and 0 <= c < COLS: rows[r][c] = 'x'
    for c, r in caves:
        if 0 <= r < ROWS and 0 <= c < COLS: rows[r][c] = 'C'
    return [''.join(row) for row in rows]


def items_value(obs):
    return {k: v for k, v in obs['items'].items() if isinstance(v, int)}


def gained(start, obs):
    before, after = items_value(start), items_value(obs)
    return (any(after.get(k, 0) > v for k, v in before.items())
            or obs['hearts']['containers'] > start['hearts']['containers'])


def changed_place(start, obs):
    return (obs['room'] != start['room'] or obs['level'] != start['level']
            or start['mode'] != 'play')


def judge(prediction, start, obs):
    """met | refuted | pending. Death refutes every prediction except 'unknown'."""
    kind = prediction['kind']
    if kind == 'unknown' or obs.get('status') != 'observed' or start.get('status') != 'observed':
        return 'pending'
    if obs['mode'] == 'death': return 'refuted'
    settled_play = obs['mode'] == 'play'
    if kind == 'reach_room':
        if settled_play and changed_place(start, obs):
            # Dungeon and overworld room ids overlap: a level change never meets it.
            same_level = obs['level'] == start['level']
            return 'met' if same_level and obs['room'] == prediction['value'] else 'refuted'
    elif kind == 'enter_cave':
        if obs['mode'] in ('cave', 'stairs') or (obs['level'] != start['level'] and obs['level'] != 0):
            return 'met'
        if settled_play and obs['room'] != start['room']: return 'refuted'
    elif kind == 'item_gained':
        if gained(start, obs): return 'met'
        if obs['mode'] in ('play', 'cave') and (obs['room'] != start['room'] or obs['mode'] != start['mode']):
            return 'refuted'
    elif kind == 'enemies_cleared':
        if obs['mode'] in ('play', 'cave') and obs['room'] == start['room'] and not obs['enemies']:
            return 'met'
        if obs['room'] != start['room']: return 'refuted'
    return 'pending'


class ZeldaGoals:
    """GoalV2 mailbox + motor for one scripted port. Same worker-facing surface as Goals."""
    INTERVAL = INTERVAL
    POLICY_ID = POLICY_ID

    def __init__(self, record, atlas=None):
        self.record = record
        self.atlas = atlas
        self.attempt = None       # the offered decision slot
        self.accepted = None      # active goal (dict, 'attempt_id' == decision_id)
        self.last_finished = None
        self.counter = 0
        self.offered_at = 0
        self.current_mask = 0
        self.latest = None
        self.until = 0  # read by Scripted.tick for applied-row attribution
        self.idle_offers = 0  # consecutive offers that expired with no goal
        self.human_intervention = False
        self.recovering = None  # set by an observed death; cleared when play returns
        self._reset_motor()

    # ---- worker/Scripted surface -------------------------------------------------
    def accepted_current(self):
        return self.accepted is not None

    def mask(self, now, seq):
        return self.current_mask if self.accepted or self.recovering else 0

    def public(self):
        return dict(attempt=self.attempt, goal=self.accepted, last_finished=self.last_finished,
                    policy_id=POLICY_ID)

    def offer(self, port, epoch, frame, observation, now):
        self.counter += 1
        view = summary(observation)
        if observation.get('status') == 'observed':
            # Only the offered decision carries terrain; status.zelda.observation stays grid-free.
            view['terrain'] = terrain(observation, self._blocked(observation),
                                      self.atlas.caves(observation['level'], observation['room'])
                                      if self.atlas and observation['mode'] == 'play' else ())
        self.attempt = dict(decision_id=f'{epoch}:{self.counter}', port=port, expected_epoch=epoch,
                            observation_frame=frame, observation=view,
                            supported_goals=list(GOALS), supported_predictions=list(PREDICTIONS),
                            previous=self.last_finished)
        self.offered_at = now
        self.record('decision_offered', **self.attempt)

    def accept(self, data, now):
        a = self.attempt
        if self.accepted and self.accepted['decision_id'] == data['decision_id']:
            if {k: v for k, v in self.accepted.items() if k in data} != data:
                raise ValueError('goal_idempotency_conflict')
            return self._public_goal()  # a retry never extends the deadline
        if not a or any(data[k] != a[k] for k in ('decision_id', 'port', 'expected_epoch')):
            raise ValueError('stale_decision')
        if self.accepted: raise ValueError('goal_active')
        if data['port'] != 0: raise ValueError('unsupported_port')  # Zelda reads controller 1 only
        if now - self.offered_at > OFFER_TTL: raise ValueError('expired_decision')
        obs = self.latest or {}
        goal, prediction = data['goal'], data['prediction']
        if goal['kind'] not in GOALS: raise ValueError('unsupported_goal')
        if prediction['kind'] not in PREDICTIONS: raise ValueError('unsupported_prediction')
        if (prediction['kind'] == 'reach_room') != (prediction.get('value') is not None):
            raise ValueError('prediction_value_mismatch')
        if goal['kind'] != 'wait' and (obs.get('status') != 'observed' or not obs['link']['controllable']):
            raise ValueError('observation_unknown')
        if (goal['kind'] == 'fight' or prediction['kind'] == 'enemies_cleared') and not obs.get('enemies'):
            raise ValueError('no_enemies_present')  # never a trivially met fight
        if goal['kind'] == 'fight' and goal['target'] != 'nearest' and not any(
                e['slot'] == goal['target'] for e in obs.get('enemies', [])):
            raise ValueError('target_not_present')
        self.accepted = dict(data, attempt_id=data['decision_id'])
        self.deadline = self.until = now + data['budget_seconds']
        self.start_obs = None; self.start_frame = None
        self.applied = 0; self.idle_offers = 0
        self._reset_motor()
        self.record('goal_committed', **data)
        return self._public_goal()

    def _recover(self, obs, frame):
        """Death recovery, only while this script grant is active: on the Continue screen,
        move the cursor (read from $13, never counted) to Continue and tap START. Each
        button is a one-frame tap, because the game acts on new presses only."""
        if obs.get('status') != 'observed': return bool(self.recovering)
        if obs['mode'] == 'death' and not self.recovering:
            self.recovering = dict(decision_id=(self.last_finished or {}).get('decision_id')
                                   if (self.last_finished or {}).get('outcome') == 'death' else None,
                                   room=obs['room'], level=obs['level'], frame=frame, last=0)
            self.attempt = None
        r = self.recovering
        if not r: return False
        menu = obs.get('menu')
        if menu and menu['kind'] == 'continue' and not menu['activated'] and r['last'] == 0:
            self.current_mask = START if menu['cursor'] == 'continue' else SELECT
        r['last'] = self.current_mask
        if obs['mode'] == 'play' and obs['link']['controllable']:
            self.record('death_recovered', decision_id=r['decision_id'], room=r['room'], level=r['level'],
                        death_frame=r['frame'], room_after=obs['room'])  # the journal stamps frame
            self.recovering = None; self.current_mask = 0
            return False  # fall through: offer a fresh decision this frame
        return True

    def note_applied(self, decision_id, mask):
        """Count only inputs the arbiter really applied for this goal (not computed masks)."""
        if self.accepted and decision_id == self.accepted['decision_id'] and mask:
            self.applied += 1

    def abort(self, reason):
        """Policy stopped (takeover, pause, sponsor loss, deadline miss, load)."""
        if reason == 'human_takeover': self.human_intervention = True
        if self.accepted:
            self._finish(f'stopped:{reason}', 'unknown', self.latest or {})
        self.attempt = None

    def observe_frame(self, port, epoch, frame, obs, now):
        """Called once per native frame after core.step. Updates the motor's next mask."""
        self.latest = obs; self.frame = frame
        if self.accepted:
            if self.start_obs is None:
                if now >= self.deadline:  # never hang on a run of unknown observations
                    return self._finish('budget_expired', 'unknown', obs)
                if obs.get('status') != 'observed': self.current_mask = 0; return
                self.start_obs = obs; self.start_frame = frame
                self.prev_hearts = obs['hearts']['current']
                self.record('goal_action_started', decision_id=self.accepted['decision_id'],
                            observation_frame=frame, observation_digest=digest(obs))
            self._evaluate(obs, now)
            if self.accepted:
                self.current_mask = self._drive(obs) if obs.get('status') == 'observed' else 0
                return
        self.current_mask = 0
        if self._recover(obs, frame): return
        stable = obs.get('status') == 'observed' and obs['link']['controllable']
        if stable and (self.attempt is None or now - self.offered_at > OFFER_TTL):
            if self.attempt is not None: self.idle_offers += 1
            self.offer(port, epoch, frame, obs, now)

    # ---- end conditions ----------------------------------------------------------
    def _evaluate(self, obs, now):
        g = self.accepted; start = self.start_obs
        verdict = judge(g['prediction'], start, obs)
        if verdict == 'met': return self._finish('prediction_met', 'met', obs)
        if obs.get('status') == 'observed' and obs['mode'] == 'death':
            return self._finish('death', 'refuted', obs)
        if verdict == 'refuted': return self._finish('prediction_refuted', 'refuted', obs)
        if self._completed(obs):
            return self._finish('completed', self._closing_verdict(), obs)
        if obs.get('status') == 'observed':
            hp = obs['hearts']['current']
            hit = hp < self.prev_hearts
            self.prev_hearts = hp
            if hit and g['goal']['kind'] not in ('fight', 'retreat'):
                return self._finish('interrupted:hit', 'unknown', obs)
        if self.still >= STUCK_FRAMES:
            return self._finish('interrupted:stuck', 'unknown', obs)
        if now >= self.deadline:
            return self._finish('budget_expired', self._closing_verdict(), obs)

    def _closing_verdict(self):
        kind = self.accepted['prediction']['kind']
        return 'met' if kind == 'survive' else 'unknown' if kind == 'unknown' else 'refuted'

    def _completed(self, obs):
        if obs.get('status') != 'observed': return False
        kind = self.accepted['goal']['kind']; start = self.start_obs
        if kind in ('exit', 'explore'):
            if obs['mode'] == 'play' and changed_place(start, obs): return True
            return kind == 'explore' and self.arrived
        if kind == 'enter':
            return obs['mode'] in ('cave', 'stairs') or obs['level'] != start['level']
        if kind == 'collect':
            # Pickup registers a few frames after touching; linger before calling it done.
            return gained(start, obs) or (self.arrived and self.tick - self.arrived_at >= COLLECT_LINGER)
        if kind == 'fight':
            if obs['room'] != start['room'] or obs['mode'] not in ('play', 'cave'): return False
            target = self.accepted['goal']['target']
            alive = {e['slot'] for e in obs['enemies']}
            return not alive if target == 'nearest' else target not in alive
        if kind == 'retreat':
            return self.retreated
        return False

    def _finish(self, outcome, verdict, obs):
        g = self.accepted
        row = dict(decision_id=g['decision_id'], goal=g['goal'], prediction=g['prediction'],
                   planner_id=g['planner_id'], outcome=outcome, verdict=verdict,
                   start_digest=digest(self.start_obs or {}), end_digest=digest(obs),
                   start_frame=getattr(self, 'start_frame', None), end_frame=getattr(self, 'frame', None),
                   applied_frames=self.applied, human_intervention=self.human_intervention,
                   bumps=self.bumps)
        self.record('goal_finished', **row)
        if self.atlas: self.atlas.decision(verdict)
        self.last_finished = row
        self.accepted = None; self.attempt = None; self.current_mask = 0; self.until = 0

    def _public_goal(self):
        return {k: v for k, v in self.accepted.items() if k != 'attempt_id'}

    # ---- motor -------------------------------------------------------------------
    def _reset_motor(self):
        self.path = None; self.plan_key = None; self.last_pos = None
        self.still = 0; self.bump_count = 0; self.bumps = 0; self.arrived = False
        self.retreated = False; self.arrived_at = 0; self.swing_at = -999; self.tick = 0; self.blocked = set()
        self.prev_invincible = 0; self.backoff = 0

    def _blocked(self, obs):
        learned = (self.atlas.blocked(obs['level'], obs['room'], obs['mode'], obs.get('walls_hash'))
                   if self.atlas else set())
        return learned | self.blocked

    def _drive(self, obs):
        self.tick += 1
        if not obs['link']['controllable'] or 'walls' not in obs:
            # Scroll, transition, cave text, item lift: the game ignores input. Hands off,
            # and never count these frames as a bump or as being stuck.
            self.last_pos = None; self.bump_count = 0; return 0
        link = obs['link']; pos = (link['x'], link['y'])
        moving = self.current_mask & (UP | DOWN | LEFT | RIGHT)
        if self.last_pos == pos and moving and self.tick - self.swing_at > SWING_COOLDOWN:
            self.still += 1; self.bump_count += 1
        else:
            if self.last_pos != pos: self.still = 0
            self.bump_count = 0
        self.last_pos = pos
        if self.bump_count >= BUMP_FRAMES:
            self._bump(obs)
        kind = self.accepted['goal']['kind']
        if kind == 'wait': return 0
        if kind in ('fight', 'retreat'): return self._combat(obs, kind)
        return self._navigate(obs, kind)

    def _bump(self, obs):
        self.bump_count = 0
        target = self.path[0] if self.path else None
        if target is None: return
        self.bumps += 1
        self.blocked.add(target)
        if self.atlas: self.atlas.mark_blocked(obs['level'], obs['room'], target, obs['mode'], obs.get('walls_hash'))
        self.record('bump', decision_id=self.accepted['decision_id'], room=obs['room'],
                    level=obs['level'], mode=obs['mode'], tile=list(target), link=[obs['link']['x'], obs['link']['y']])
        self.path = None  # replan around it

    def _targets(self, obs, kind, here):
        walls, blocked = obs['walls'], self._blocked(obs)
        goal = self.accepted['goal']
        if kind in ('exit', 'explore'):
            tiles, _ = edge_tiles(walls, blocked, here, goal['dir'])
            return tiles, goal['dir']
        tx, ty = goal['tile']['x'], goal['tile']['y']
        if kind == 'enter':
            # A cave or stair tile is "unwalkable"; walking into it from below enters it.
            cands = [(c, ty + 1) for c in (tx - 1, tx) if open_tile(walls, blocked, c, ty + 1)]
            return cands, 'N'
        # collect: stand on the tile itself; only if Link cannot, stand beside it.
        if open_tile(walls, blocked, tx, ty): return [(tx, ty)], None
        cands = [(c, r) for c in (tx - 1, tx, tx + 1) for r in (ty - 1, ty, ty + 1)
                 if open_tile(walls, blocked, c, r)]
        return cands, None

    def _navigate(self, obs, kind):
        here = tile_of(obs['link']['x'], obs['link']['y'])
        key = (obs['room'], obs['mode'], obs['walls_hash'])
        if self.path is None or self.plan_key != key:
            targets, push = self._targets(obs, kind, here)
            self.push = push
            self.goal_tiles = set(targets)
            self.path = astar(obs['walls'], self._blocked(obs), here, targets)
            self.plan_key = key
            if self.path is None:
                self.record('no_path', decision_id=self.accepted['decision_id'], room=obs['room'],
                            tile=list(here))
                self.path = []
                self.still = STUCK_FRAMES  # nothing reachable: end now as stuck, not silently
                return 0
        return self._follow(obs, here)

    def _follow(self, obs, here):
        x, y = obs['link']['x'], obs['link']['y']
        while self.path and near(x, y, *tile_xy(*self.path[0])):
            self.path.pop(0)
        if self.path:
            return self._toward(x, y, *tile_xy(*self.path[0]))
        tx, ty = tile_xy(*here)
        if here in self.goal_tiles:
            # The last tile is approached exactly: pickups need real overlap (live:
            # stopping 2 px, or 1 px from the left, short of the cave sword never took it).
            if tx != x: return RIGHT if tx > x else LEFT
            if ty != y: return DOWN if ty > y else UP
        kind = self.accepted['goal']['kind']
        if kind == 'explore' and not on_screen_edge(here, self.push):
            self.arrived = True; return 0
        if kind == 'collect':
            if not self.arrived: self.arrived, self.arrived_at = True, self.tick
            return 0
        return DIRS[self.push][2] if self.push else 0  # push through the exit / into the cave

    @staticmethod
    def _toward(x, y, tx, ty):
        if abs(tx - x) > SLACK: return RIGHT if tx > x else LEFT
        if abs(ty - y) > SLACK: return DOWN if ty > y else UP
        return 0

    # ---- combat reflex (fight / retreat only) -----------------------------------
    def _combat(self, obs, kind):
        link = obs['link']; x, y = link['x'], link['y']
        foes = obs['enemies']
        if not foes:
            if kind == 'retreat': self.retreated = True
            return 0
        target = self._target(obs, foes)
        dist = abs(target['x'] - x) + abs(target['y'] - y)
        inv = link['invincible_frames']
        if inv > self.prev_invincible: self.backoff = 12  # just hit: back off briefly
        self.prev_invincible = inv
        if kind == 'retreat' and dist >= RETREAT_DISTANCE:
            self.retreated = True; return 0
        swing = self._swing(obs, target)
        if swing is not None and kind == 'fight' and not self.backoff:
            return swing
        if self.backoff or kind == 'retreat':
            self.backoff = max(0, self.backoff - 1)
            if swing is not None and dist < NEAR: return swing  # cornered: strike
            return self._away(obs, target)
        return self._approach(obs, target)

    def _target(self, obs, foes):
        goal = self.accepted['goal']; x, y = obs['link']['x'], obs['link']['y']
        if goal['kind'] == 'fight' and goal['target'] != 'nearest':
            hit = [e for e in foes if e['slot'] == goal['target']]
            if hit: return hit[0]
        return min(foes, key=lambda e: abs(e['x'] - x) + abs(e['y'] - y))

    def _swing(self, obs, e):
        """Face and swing when the target is on Link's row or column and in reach."""
        link = obs['link']; x, y = link['x'], link['y']
        dx, dy = e['x'] - x, e['y'] - y
        full = obs['hearts']['current'] >= obs['hearts']['containers']
        reach = 256 if full and obs['items'].get('sword') else NEAR
        if abs(dy) <= 6 and abs(dx) <= reach: face = 'E' if dx > 0 else 'W'
        elif abs(dx) <= 6 and abs(dy) <= reach: face = 'S' if dy > 0 else 'N'
        else: return None
        if link['facing'] != face: return DIRS[face][2]  # one-frame turn
        if self.tick - self.swing_at >= SWING_COOLDOWN and not (self.current_mask & A):
            self.swing_at = self.tick
            return A
        return 0

    def _approach(self, obs, e):
        walls, blocked = obs['walls'], self._blocked(obs)
        here = tile_of(obs['link']['x'], obs['link']['y'])
        ec, er = tile_of(e['x'], e['y'])
        if self.path is None or self.tick % 15 == 0:
            cands = [(c, er) for c in range(0, MAX_COL + 1) if 2 <= abs(c - ec) <= 6]
            cands += [(ec, r) for r in range(MIN_ROW, MAX_ROW + 1) if 2 <= abs(r - er) <= 5]
            cands = [t for t in cands if open_tile(walls, blocked, *t)]
            self.path = astar(walls, blocked, here, cands) or []
            self.goal_tiles = set(cands); self.push = None
        x, y = obs['link']['x'], obs['link']['y']
        while self.path and near(x, y, *tile_xy(*self.path[0])): self.path.pop(0)
        if self.path: return self._toward(x, y, *tile_xy(*self.path[0]))
        # Aligned tile reached (or none reachable): close the last pixels directly.
        dx, dy = e['x'] - x, e['y'] - y
        if abs(dx) < abs(dy): return self._toward(x, y, e['x'], y) or self._toward(x, y, x, e['y'])
        return self._toward(x, y, x, e['y']) or self._toward(x, y, e['x'], y)

    def _away(self, obs, e):
        walls, blocked = obs['walls'], self._blocked(obs)
        here = tile_of(obs['link']['x'], obs['link']['y'])
        ec, er = tile_of(e['x'], e['y'])
        area = reachable(walls, blocked, here, limit=200)
        best = max(area | {here}, key=lambda t: abs(t[0] - ec) + abs(t[1] - er))
        path = astar(walls, blocked, here, [best]) or []
        self.path = path
        if not path: return 0
        return self._toward(obs['link']['x'], obs['link']['y'], *tile_xy(*path[0]))
