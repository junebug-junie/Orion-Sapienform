"""Explicitly handed-off, open-loop baseline. No perception or learning claim."""

from .goals import Goals, POLICY_ID as GOAL_POLICY_ID

POLICY_ID = 'scripted-right-jump-v1'
INTERVAL = 1 / 15
HOLD_TTL = .1


class Scripted:
    def __init__(self, authority, record):
        self.authority = authority
        self.record = record
        self.active = {}

    def public(self):
        return [dict(port=p, policy_id=s['policy_id'], epoch=s['grant']['epoch'],
                     duration_seconds=s['duration'], started_frame=s['frame'],
                     cognition=s['goals'].public() if s['goals'] else None)
                for p, s in sorted(self.active.items())]

    def start(self, sponsor, port, duration, now, frame, expected_epoch, cognition=False):
        old = self.authority.leases.get(port)
        # Explicit transfer of the caller's own live controller only. Scripts
        # cannot seize another human's port, even through the shared Hub identity.
        if (not old or old['connection'] != sponsor or old['expires'] <= now
                or old['epoch'] != expected_epoch):
            raise ValueError('script_requires_owned_controller')
        connection = 'scripted:'+str(port)
        grant = self.authority.control(connection, port, 'takeover', now)
        self.active[port] = dict(sponsor=sponsor, connection=connection, grant=grant,
                                 duration=duration, until=now+duration, next=now,
                                 seq=0, frame=frame, goal_attempt_id=None, goals=Goals(self.record) if cognition else None,
                                 policy_id=GOAL_POLICY_ID if cognition else POLICY_ID)
        self.record('policy_started', port=port, policy_id=self.active[port]['policy_id'],
                    epoch=grant['epoch'], duration_seconds=duration)

    def stop(self, port, reason):
        s = self.active.pop(port, None)
        if not s:
            return
        # A human may already have taken over. Never release their newer lease.
        self.authority.disconnect(s['connection'])
        self.record('policy_stopped', port=port, policy_id=s['policy_id'],
                    epoch=s['grant']['epoch'], reason=reason)

    def cancel(self, reason, sponsor=None):
        for port, s in list(self.active.items()):
            if sponsor is None or sponsor == s['sponsor']:
                self.stop(port, reason)

    def tick(self, now):
        for port, s in list(self.active.items()):
            lease = self.authority.leases.get(port)
            if not lease or lease['connection'] != s['connection']:
                self.stop(port, 'human_takeover')
            elif now >= s['until']:
                self.stop(port, 'budget_expired')
            elif now >= lease['expires']:
                self.stop(port, 'policy_deadline_missed')
            elif now >= s['next']:
                # Four decisions jumping right, eleven walking right. A fixed
                # diagnostic sequence, independent of pixels, RAM, or outcomes.
                mask = (s['goals'].mask(now, s['seq']) if s['goals'] else
                        128 | (256 if s['seq'] % 15 < 4 else 0))
                goals = s['goals']
                s['goal_attempt_id'] = (goals.accepted['attempt_id']
                    if goals and goals.accepted_current() and now < goals.until else None)
                s['seq'] += 1
                self.authority.submit(s['connection'],
                    dict(s['grant'], seq=s['seq'], mask=mask), now)
                lease['expires'] = now+HOLD_TTL
                s['next'] = now+INTERVAL  # no catch-up burst after a stall

    def annotate(self, event):
        s = self.active.get(event['port'])
        if s and event['epoch'] == s['grant']['epoch']:
            event['policy_id'] = s['policy_id']
            if s['goal_attempt_id'] and event['seq'] == s['seq']:
                event['goal_attempt_id'] = s['goal_attempt_id']

    def goal(self, sponsor, data, now):
        s = self.active.get(data['port'])
        lease = self.authority.leases.get(data['port'])
        if (not s or not s['goals'] or s['sponsor'] != sponsor or not lease
                or now >= s['until'] or lease['connection'] != s['connection'] or lease['expires'] <= now
                or s['grant']['epoch'] != data['expected_epoch']):
            raise ValueError('goal_requires_active_controller')
        return s['goals'].accept(data, now)
