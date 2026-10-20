"""Only this spawned process loads the native core. Commands never call retro_run."""
import hashlib
from pathlib import Path
import queue
import time
from orion.retro.probe import CoreProbe
from .media import MediaProbe
from .control import Arbiter
from .evidence import Evidence
from .scripted import Scripted
from .observation import Observer, outcome


def run(commands, urgent, replies, stop, settings, game, session, generation, media=None):
    core = evidence = None
    state = 'starting'; frame = 0
    def send(row):
        replies.put_nowait(row)  # backpressure is fatal, never unbounded
    try:
        core = (MediaProbe if media is not None else CoreProbe)(settings.core, game['path'])
        evidence = Evidence(settings.artifacts, session, settings.session_bytes, settings.total_bytes)
        authority = Arbiter(game['ports'])
        def record(kind, **fields):
            evidence.write(dict(kind=kind, session_id=session, generation=generation, frame=frame, **fields))
        scripted = Scripted(authority, record)
        observer = Observer(core)
        sources = sorted(Path(__file__).parent.glob('*.py')) + [Path(__file__).parents[1]/'settings.py']
        runtime_sha = hashlib.sha256(b''.join(p.name.encode()+b'\0'+p.read_bytes() for p in sources)).hexdigest()
        record('start', game_id=game['game_id'], identity=core.identity, runtime_sha256=runtime_sha)
        state = 'running'; paused_by = ''
        send(dict(kind='ready', identity=core.identity, runtime_sha256=runtime_sha, fps=core.av.timing.fps))
        period = 1 / core.av.timing.fps
        deadline = time.monotonic(); last_status = deadline; misses = 0
        acquire_deadline = deadline+5
        pending = {}
        def not_running():
            # Say why play stopped, so a client can tell an operator Pause from a
            # lapsed lease or a dropped controller, which it may heal.
            return 'session_paused:'+paused_by if state == 'paused' else 'session_not_running'
        def reject_pending(reason, ports=None):
            for key in list(pending):
                if ports is None or key in ports:
                    send(dict(kind='reply', id=pending.pop(key), error=reason))
        while not stop.is_set():
            # Work per frame is bounded even when an ingress floods the pipe.
            for _ in range(16):
                try: cmd = urgent.get_nowait()
                except queue.Empty:
                    try: cmd = commands.get_nowait()
                    except queue.Empty: break
                request = cmd['id']; op = cmd['op']; data = cmd.get('data', {})
                try:
                    if op == 'input':
                        if state != 'running': raise ValueError(not_running())
                        if time.monotonic()-cmd['received_at'] >= .25: raise ValueError('expired_in_queue')
                        authority.submit(cmd['connection'], data, cmd['received_at'])
                        key = data['port']
                        if key in pending:
                            send(dict(kind='reply', id=pending[key], error='superseded'))
                        pending[key] = request
                        continue
                    if op == 'control':
                        if state != 'running': raise ValueError(not_running())
                        result = authority.control(cmd['connection'], **data, now=time.monotonic())
                        if data['port'] in pending:
                            send(dict(kind='reply', id=pending.pop(data['port']), error='lease_replaced'))
                        record('control', port=data['port'], action=data['action'])
                        scripted.tick(time.monotonic())
                    elif op == 'scripted':
                        if state != 'running': raise ValueError(not_running())
                        port = data['port']
                        if data['action'] == 'start':
                            if data.get('cognition') and not settings.cognition_enabled:
                                raise ValueError('retro_cognition_disabled')
                            scripted.start(cmd['connection'], port, data['duration_seconds'],
                                           time.monotonic(), frame, data['expected_epoch'], data.get('cognition', False))
                            reject_pending('control_revoked', [port])
                        else:
                            active = scripted.active.get(port)
                            if active and active['grant']['epoch'] != data['expected_epoch']:
                                raise ValueError('stale_script_epoch')
                            scripted.stop(port, 'operator_stop')
                        result = {'scripted': scripted.public()}
                    elif op == 'goal':
                        if state != 'running': raise ValueError(not_running())
                        result = scripted.goal(cmd['connection'], data, time.monotonic())
                    elif op in ('pause', 'resume', 'disconnect'):
                        if op == 'disconnect':
                            scripted.cancel('sponsor_disconnected', cmd['connection'])
                            released = authority.disconnect(cmd['connection'])
                            reject_pending('control_revoked', released)
                            if released and not authority.leases: state = 'paused'; paused_by = 'disconnect'
                        elif op == 'pause' and state == 'paused':
                            scripted.cancel('pause')
                            paused_by = 'pause'  # an explicit Pause makes an auto-pause stick
                        elif (op == 'pause' and state != 'paused') or (op == 'resume' and state != 'running'):
                            scripted.cancel(op)
                            authority.clear()
                            reject_pending('control_revoked')
                            state = 'paused' if op == 'pause' else 'running'; paused_by = op
                            if op == 'resume':
                                # Paused ports are already neutral; don't mistake their old
                                # release barrier for a fresh lease loss after resume.
                                authority.neutral.clear()
                                acquire_deadline = time.monotonic()+5
                        if state == 'paused': core.buttons = (0, 0)
                        record('transition', state=state, reason=op)
                        result = {'state': state}
                        deadline = time.monotonic()
                    else: raise ValueError('unknown_command')
                    send(dict(kind='reply', id=request, result=result))
                except ValueError as exc:
                    send(dict(kind='reply', id=request, error=str(exc)))
            if state == 'running':
                scripted.tick(time.monotonic())
                masks, applied, neutral = authority.frame(time.monotonic())
                for event in applied:
                    policy = scripted.active.get(event['port'])
                    goals = policy['goals'] if policy else None
                    if goals and goals.accepted_current() and time.monotonic() < goals.until:
                        goals.applied_start(frame+1, observer.read())
                core.step(*masks); frame += 1
                for port, policy in scripted.active.items():
                    goals = policy['goals']
                    if not goals: continue
                    now = time.monotonic()
                    if goals.ready_for_observation(now):
                        observation = observer.read()
                        previous = goals.start_observation if goals.accepted_current() else {}
                        goal = goals.accepted['goal'] if goals.accepted_current() else 'wait'
                        verdict = outcome(previous or {}, observation, goal)
                        goals.offer(port, policy['grant']['epoch'], frame, observation, now, verdict)

                if media is not None: core.publish(media, frame)
                if neutral: record('neutral', ports=neutral)
                if not authority.leases and (neutral or time.monotonic() >= acquire_deadline):
                    state = 'paused'; paused_by = 'no_controller'
                    record('transition', state=state, reason='no_controller')
                for event in applied:
                    scripted.annotate(event)
                    record('applied', **event)
                    if 'policy_id' in event:
                        send(dict(kind='applied', **event, frame=frame))
                    request = pending.pop(event['port'], None)
                    if request: send(dict(kind='reply', id=request, result=dict(event, frame=frame)))
                for port in list(pending):
                    if port not in authority.leases:
                        send(dict(kind='reply', id=pending.pop(port), error='expired_before_frame'))
                deadline += period
                now = time.monotonic()
                late = now > deadline
                misses = misses+1 if late else 0
                if late: deadline = now  # never execute catch-up bursts
                if misses >= 3:
                    scripted.cancel('deadline_overrun')
                    state = 'paused'; paused_by = 'deadline_overrun'; authority.clear(); reject_pending('deadline_overrun')
                    core.buttons = (0, 0)
                    record('transition', state=state, reason='deadline_overrun')
                time.sleep(max(0, deadline-time.monotonic()))
            else:
                time.sleep(.01)
            now = time.monotonic()
            if now-last_status >= .1:
                send(dict(kind='status', state=state, frame=frame, video_frames=core.video_frames,
                          audio_samples=core.audio_samples, frame_hash=core.frame_hash,
                          scripted=scripted.public()))
                last_status = now
        scripted.cancel('worker_stop')
        authority.clear(); core.buttons = (0, 0)
        reject_pending('stopped')
        record('transition', state='stopped', reason='operator_or_service_stop')
    except BaseException as exc:
        try: send(dict(kind='fatal', error=type(exc).__name__+': '+str(exc)[:200]))
        except queue.Full: pass
    finally:
        if core: core.close()
        if evidence: evidence.close()
