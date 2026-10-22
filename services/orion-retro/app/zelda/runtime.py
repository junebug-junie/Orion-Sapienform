"""Per-frame Zelda hook shared by the native worker and the native eval."""
import time
from .observe import ZeldaObserver, summary
from .navigate import ZeldaGoals
from .atlas import Atlas, LINEAGE
from ..saves import available


class ZeldaRuntime:
    def __init__(self, core, saves_root, session_id, record, lineage_id=None, now=time.monotonic):
        self.observer = ZeldaObserver(core)
        self.supported = self.observer.supported
        self.record = record; self.latest = {'status': 'unknown', 'reason': 'not_read'}
        self.atlas = None
        if self.supported and available(saves_root):
            self.atlas = Atlas(saves_root, lineage_id or Atlas.new_lineage(), core.identity,
                               session_id, now=now, record=record)
            record('atlas_opened', lineage_id=self.atlas.lineage_id)

    def goals(self):
        return ZeldaGoals(self.record, self.atlas)

    def after_frame(self, frame, now, active):
        obs = self.observer.read(frame)
        self.latest = obs
        if self.atlas:
            self.atlas.observe(obs, frame)
            self.atlas.flush()
        for port, policy in active.items():
            goals = policy.get('goals')
            if isinstance(goals, ZeldaGoals):
                goals.observe_frame(port, policy['grant']['epoch'], frame, obs, now)
        return obs

    def lineage_id(self):
        return self.atlas.lineage_id if self.atlas else None

    def on_load(self, lineage_id, initial=False):
        if not self.atlas: return
        if not (isinstance(lineage_id, str) and LINEAGE.match(lineage_id)): lineage_id = None
        if lineage_id and lineage_id != self.atlas.lineage_id or (not lineage_id and not initial):
            self.atlas.open(lineage_id or Atlas.new_lineage())
            self.record('atlas_opened', lineage_id=self.atlas.lineage_id)
        else:
            self.atlas.prev = None; self.atlas.place = None  # never link rooms across a load

    def status(self):
        return dict(observation=summary(self.latest), atlas=self.atlas.public() if self.atlas else None)

    def close(self):
        if self.atlas:
            self.atlas.end_session(); self.atlas.flush(force=True)
