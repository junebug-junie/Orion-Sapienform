"""Bounded goal mailbox; only the native worker can accept and actuate a goal."""
GOALS = ('wait', 'walk_left', 'walk_right', 'jump_left', 'jump_right')
POLICY_ID = 'goal-motor-v1'


class Goals:
    def __init__(self, record):
        self.record = record
        self.attempt = None
        self.accepted = None
        self.until = 0
        self.counter = 0
        self.start_observation = None
        self.start_frame = None

    def offer(self, port, epoch, frame, observation, now, outcome='unknown'):
        self.counter += 1
        self.attempt = dict(attempt_id=f'{epoch}:{self.counter}', port=port,
                            expected_epoch=epoch, observation_frame=frame, observation=observation,
                            outcome=outcome, supported_goals=list(GOALS),
                            previous_goal=self.accepted['goal'] if self.accepted else 'wait')
        self.offered_at = now
        self.record('attempt_finished', **self.attempt)

    def accept(self, data, now):
        a = self.attempt
        if not a or any(data[k] != a[k] for k in ('attempt_id', 'port', 'expected_epoch')):
            raise ValueError('stale_attempt')
        if now-self.offered_at > 30:
            raise ValueError('expired_attempt')
        if data['goal'] not in GOALS:
            raise ValueError('unsupported_goal')
        if self.accepted and self.accepted['attempt_id'] == data['attempt_id']:
            if self.accepted != data:
                raise ValueError('goal_idempotency_conflict')
            return dict(self.accepted)  # retry never extends the deadline
        self.accepted = dict(data)
        self.until = now+5
        self.start_observation = None
        self.start_frame = None
        self.record('goal_committed', **data, duration_seconds=5)
        return dict(data)

    def applied_start(self, frame, observation):
        if self.start_observation is None:
            self.start_observation = observation
            self.start_frame = frame
            self.record('goal_action_started', attempt_id=self.accepted['attempt_id'],
                        observation_frame=frame, observation=observation)

    def accepted_current(self):
        return bool(self.accepted and self.attempt
                    and self.accepted['attempt_id'] == self.attempt['attempt_id'])

    def ready_for_observation(self, now):
        if self.attempt is None: return True
        return now >= self.until if self.accepted_current() else now-self.offered_at > 30

    def mask(self, now, seq):
        goal = self.accepted['goal'] if self.accepted and now < self.until else 'wait'
        direction = 64 if goal.endswith('left') else 128
        return 0 if goal == 'wait' else direction | (256 if goal.startswith('jump') and seq % 15 < 4 else 0)

    def public(self):
        return dict(attempt=self.attempt, goal=self.accepted)
