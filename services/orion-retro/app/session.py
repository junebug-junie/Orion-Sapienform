"""Async API-side lifecycle. All native calls stay in the spawned child."""
import asyncio
from collections import deque
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import uuid
from .worker import run
from .media import MediaSlot
from .saves import SaveStore, available
from orion.retro.probe import digest


# Startup hashes every ROM once; NES ROMs are small, so 64 stays well under a second.
MAX_GAMES = 64


class Sessions:
    def __init__(self, settings):
        self.settings = settings
        self.generation = str(uuid.uuid4())
        self.catalog = {}; self.session = None; self.process = None
        self.pending = {}; self.connections = set(); self.history = deque(maxlen=256)
        self.media = None; self.media_task = None; self.latest_media = None; self.media_version = 0
        self.requests = {}; self.task = None; self.stopping = False; self.starting_save = None
        self.ready = asyncio.Event()
        self.settings.artifacts.mkdir(parents=True, exist_ok=True)
        self.status_path = self.settings.artifacts/'status.json'
        if self.status_path.exists():
            self.session = json.loads(self.status_path.read_text())
            if self.session['state'] in ('starting', 'running', 'paused'):
                self.session.update(state='failed', reason='service_restart', scripted=[])
                self.persist()
        catalog = json.loads(settings.catalog.read_text())
        if not isinstance(catalog, list) or not 1 <= len(catalog) <= MAX_GAMES:
            raise ValueError(f'catalog must contain 1..{MAX_GAMES} games')
        for item in catalog:
            if set(item) != {'game_id','title','file','rom_sha256','ports','provenance'}:
                raise ValueError('invalid catalog fields')
            from .contracts import Start
            Start(game_id=item['game_id'], idempotency_key='catalog')
            path = (settings.catalog.parent/item['file']).resolve(strict=True)
            if not path.is_relative_to(settings.catalog.parent.resolve()) or path.suffix != '.nes':
                raise ValueError('ROM must be inside mounted catalog')
            if not 0 < path.stat().st_size <= 32*1024*1024:
                raise ValueError('invalid ROM size')
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['rom_sha256']:
                raise ValueError('ROM digest mismatch')
            if item['ports'] not in ([0], [0,1]) or item['game_id'] in self.catalog:
                raise ValueError('invalid ports or duplicate game')
            if not isinstance(item['title'], str) or len(item['title']) > 128:
                raise ValueError('invalid title')
            self.catalog[item['game_id']] = dict(item, path=path)
        if not settings.core.is_file(): raise ValueError('core missing')
        # Read the file only; the native core is loaded solely in the worker child.
        self.core_sha256 = digest(settings.core)
        self.saves = SaveStore(settings.saves, settings.save_max_per_game, settings.save_max_bytes)

    def save_enabled(self):
        return available(self.settings.saves)

    def identity(self, game_id):
        return dict(rom_sha256=self.catalog[game_id]['rom_sha256'], core_sha256=self.core_sha256)

    def list_saves(self, game_id):
        if game_id not in self.catalog: raise ValueError('unknown_game')
        if not self.save_enabled(): return []
        identity = self.identity(game_id)
        return [dict(m, compatible=m['rom_sha256'] == identity['rom_sha256']
                     and m['core_sha256'] == identity['core_sha256'])
                for m in self.saves.list(game_id)]

    def games(self):
        return [dict(game_id=v['game_id'], title=v['title'], ports=v['ports'],
                     rom_sha256=v['rom_sha256'], save_enabled=self.save_enabled())
                for v in self.catalog.values()]

    def persist(self):
        tmp = self.status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.session)+'\n')
        tmp.replace(self.status_path)

    async def start(self, game_id, key, owner="operator", save_id=None):
        key = owner+":"+key
        if key in self.requests:
            old_game, old_id = self.requests[key]
            if old_game != (game_id, save_id): raise ValueError('idempotency_conflict')
            return {'session_id': old_id}
        if game_id not in self.catalog: raise ValueError('unknown_game')
        if self.process is not None or self.stopping: raise ValueError('session_active')
        if save_id is not None:
            if not self.save_enabled(): raise ValueError('saves_unavailable')
            self.saves.meta(game_id, save_id, self.identity(game_id))  # refuse before spawning
        if len(self.requests) >= 256: raise ValueError('idempotency_capacity_restart_required')
        ctx = mp.get_context('spawn')
        self.commands = ctx.Queue(32); self.urgent = ctx.Queue(8); self.replies = ctx.Queue(128); self.stop_event = ctx.Event()
        self.media = MediaSlot(ctx); self.latest_media = None; self.media_version = 0
        self.ready = asyncio.Event()
        sid = str(uuid.uuid4())
        self.session = dict(session_id=sid, generation=self.generation, state='starting',
                            game_id=game_id, frame=0, owner=owner, save_enabled=self.save_enabled(),
                            loaded_save=None, scripted=[], game_profile=None)
        self.starting_save = save_id
        self.persist(); self.history.clear()
        self.process = ctx.Process(target=run, args=(self.commands,self.urgent,self.replies,self.stop_event,
                            self.settings,dict(self.catalog[game_id], save_id=save_id),sid,self.generation,self.media), daemon=True)
        self.process.start(); self.requests[key] = ((game_id, save_id), sid)
        self.task = asyncio.create_task(self.monitor())
        self.media_task = asyncio.create_task(self.monitor_media())
        return dict(self.session)

    async def monitor_media(self):
        from .media import compress
        try:
            while self.process:
                packet = None
                for _ in range(2):
                    try: packet = self.media.get_nowait()
                    except queue.Empty: break
                if packet is not None:
                    self.latest_media = await asyncio.to_thread(compress, packet)
                    self.media_version += 1
                await asyncio.sleep(.005)
        except asyncio.CancelledError:
            pass

    async def monitor(self):
        last = time.monotonic(); started = last; ready = False
        try:
            while self.process:
                for _ in range(128):
                    try: row = self.replies.get_nowait()
                    except queue.Empty: break
                    last = time.monotonic(); kind = row['kind']
                    if kind == 'reply':
                        future = self.pending.pop(row['id'], None)
                        if future and not future.done(): future.set_result(row)
                        if 'frame' in row.get('result', {}): self.history.append(row['result'])
                    elif kind == 'ready':
                        # The worker loads a start save before it reports ready.
                        ready = True; self.ready.set(); self.session.update(state='running', identity=row['identity'], runtime_sha256=row['runtime_sha256'], fps=row['fps'], loaded_save=self.starting_save, game_profile=row.get('game_profile'))
                        self.persist()
                    elif kind == 'status':
                        self.session.update({k:v for k,v in row.items() if k!='kind'})
                    elif kind == 'applied':
                        self.history.append({k:v for k,v in row.items() if k!='kind'})
                    elif kind == 'fatal':
                        await self.stop('failed', row['error'], from_monitor=True); return
                if not self.process.is_alive():
                    await self.stop('failed', 'worker_exited', from_monitor=True); return
                if time.monotonic()-(last if ready else started) > (1 if ready else 10):
                    await self.stop('failed', 'worker_timeout', from_monitor=True); return
                await asyncio.sleep(.005)
        except asyncio.CancelledError:
            pass
        except Exception:
            await self.stop('failed', 'monitor_failure', from_monitor=True)

    async def command(self, op, connection='', data=None):
        if not self.process or self.stopping or len(self.pending) >= 24: raise ValueError('unavailable_or_busy')
        if not self.ready.is_set():
            # A command sent right after start (the browser's resume) must not spend its
            # 0.75 s budget on core loading (~0.5 s live), or the session fails.
            # 4.5 s stays under Hub's 5 s proxy timeout; monitor() fails a worker not
            # ready within 10 s. Re-check after the wait: waiters are not yet in
            # pending, so the cap (24, below the 32-slot queue) must be enforced
            # again, and a restart must not hand an old waiter the new session.
            sid = (self.session or {}).get('session_id')
            try: await asyncio.wait_for(self.ready.wait(), 4.5)
            except asyncio.TimeoutError: raise ValueError('worker_unavailable')
            if (not self.process or self.stopping or len(self.pending) >= 24
                    or (self.session or {}).get('session_id') != sid):
                raise ValueError('unavailable_or_busy')
        rid = str(uuid.uuid4()); future = asyncio.get_running_loop().create_future()
        self.pending[rid] = future
        try:
            pipe = self.commands if op == 'input' else self.urgent
            pipe.put_nowait(dict(id=rid, op=op, connection=connection, data=data or {}, received_at=time.monotonic()))
            row = await asyncio.wait_for(future, .75)
            if 'error' in row: raise ValueError(row['error'])
            if 'state' in row['result']:
                self.session['state'] = row['result']['state']; self.persist()
            if 'loaded' in row['result']:
                self.session['loaded_save'] = row['result']['loaded']['save_id']; self.persist()
            return row['result']
        except (asyncio.TimeoutError, queue.Full):
            # Shielded: a cancelled caller (closed input socket) must not abandon cleanup.
            await asyncio.shield(self.stop('failed', 'command_timeout_or_overload'))
            raise ValueError('worker_unavailable')
        finally:
            self.pending.pop(rid, None)

    async def stop(self, state='stopped', reason='operator_stop', from_monitor=False):
        process, self.process = self.process, None
        if not process: return
        self.stopping = True
        self.ready.set()  # wake start-time waiters so they fail now, not after their timeout
        if self.task and not from_monitor: self.task.cancel()
        if self.media_task: self.media_task.cancel()
        self.latest_media = None
        self.stop_event.set()
        await asyncio.to_thread(process.join, .3)
        if process.is_alive():
            process.terminate(); await asyncio.to_thread(process.join, .3)
        if process.is_alive():
            process.kill(); await asyncio.to_thread(process.join, .3)
        for future in self.pending.values():
            if not future.done(): future.set_result({'error':'session_stopped'})
        self.pending.clear(); self.connections.clear()
        self.commands.cancel_join_thread(); self.commands.close()
        self.urgent.cancel_join_thread(); self.urgent.close()
        self.replies.cancel_join_thread(); self.replies.close()
        self.media = None  # A dead writer's lock may be abandoned; never reuse it.
        self.session.update(state=state, reason=reason, scripted=[])
        try:
            self.persist()
        finally:
            self.stopping = False

    async def disconnect(self, connection, sid, generation):
        owned = connection in self.connections
        self.connections.discard(connection)
        if (owned and self.process and self.session and self.session['session_id'] == sid
                and self.generation == generation):
            try: await self.command('disconnect', connection)
            except ValueError: pass

    def require(self, sid):
        if not self.session or self.session['session_id'] != sid:
            raise ValueError('unknown_session')
