import ctypes as C
from types import SimpleNamespace

import pytest

from orion.retro.preflight import benchmark
from orion.retro.probe import CoreProbe, Variable


class Clock:
    now = 0.
    def __call__(self):
        return self.now
    def sleep(self, amount):
        self.now += amount


class FakeCore:
    identity = {"core_sha256": "test-only"}
    unsupported = set()
    av = SimpleNamespace(timing=SimpleNamespace(fps=60., sample_rate=48000))
    def __init__(self, clock, *, delay=.001, video=True, audio=True, corrupt=False):
        self.clock, self.delay = clock, delay
        self.video, self.audio, self.corrupt = video, audio, corrupt
        self.video_frames = self.audio_samples = self.state = 0
        self.frame_hash = None
    def step(self, p1=0, p2=0):
        self.clock.now += self.delay
        self.video_frames += self.video
        self.audio_samples += 800 * self.audio
        self.state += p1 + 2 * p2 + 1
        self.frame_hash = str(self.state) if self.video else None
        return self.frame_hash
    def snapshot(self):
        return str(self.state).encode()
    def restore(self, state):
        self.state = int(state) + (self.video_frames if self.corrupt else 0)


def run(**kwargs):
    clock = Clock()
    return benchmark(FakeCore(clock, **kwargs), 120, 1000, clock=clock, sleep=clock.sleep)


def test_short_run_cannot_claim_acceptance():
    result = run()
    assert result["status"] == "INCOMPLETE_OR_FAILED"
    assert result["gates"]["same_core_replay"]
    assert not result["gates"]["thirty_minute_sample"]
    assert result["gates"]["nominal_cadence"]


@pytest.mark.parametrize("kwargs,gate", [
    ({"delay": .02}, "nominal_cadence"),
    ({"video": False}, "video_observed"),
    ({"audio": False}, "audio_observed"),
    ({"corrupt": True}, "same_core_replay"),
])
def test_failed_evidence_cannot_pass(kwargs, gate):
    assert not run(**kwargs)["gates"][gate]


def bare_probe():
    probe = CoreProbe.__new__(CoreProbe)
    probe.buttons = (1 << 8, 1 << 0)
    probe.options, probe.unsupported = {}, set()
    probe.pixel_format = probe.video_frames = 0
    probe.frame_hash = probe.error = None
    return probe


def test_ports_are_independent_and_unknown_inputs_neutral():
    probe = bare_probe()
    assert probe._input(0, 1, 0, 8) == 1
    assert probe._input(1, 1, 0, 8) == 0
    assert probe._input(1, 1, 0, 0) == 1
    for request in [(2, 1, 0, 0), (0, 3, 0, 8), (0, 1, 1, 8), (0, 1, 0, 256)]:
        assert probe._input(*request) == 0


def test_video_padding_is_excluded_and_duplicates_keep_hash():
    probe = bare_probe()
    a = C.create_string_buffer(b"1234xx5678xx")
    b = C.create_string_buffer(b"1234zz5678zz")
    probe._video(C.addressof(a), 2, 2, 6)
    expected = probe.frame_hash
    probe._video(C.addressof(b), 2, 2, 6)
    assert probe.frame_hash == expected
    probe._video(None, 2, 2, 6)
    assert probe.frame_hash == expected
    probe._video(C.addressof(a), 2, 2, 1)
    assert probe.error == "invalid video geometry"


def test_environment_defaults_and_unsupported_commands():
    probe = bare_probe()
    variables = (Variable * 2)(Variable(b"region", b"Region; ntsc|pal"), Variable())
    assert probe._environment(16, C.addressof(variables))
    query = Variable(b"region", None)
    assert probe._environment(15, C.addressof(query))
    assert query.value == b"ntsc"
    assert not probe._environment(999, None)
    assert 999 in probe.unsupported


def test_invalid_frame_count():
    with pytest.raises(ValueError):
        benchmark(None, 0, 1)


def test_consistently_broken_restore_cannot_pass():
    class BrokenRestore(FakeCore):
        def restore(self, state):
            self.state = 0
    clock = Clock()
    result = benchmark(BrokenRestore(clock), 108000, 1000,
                       clock=clock, sleep=clock.sleep)
    assert not result["gates"]["same_core_replay"]
    assert result["status"] == "INCOMPLETE_OR_FAILED"


def test_first_replay_frame_can_duplicate_checkpoint_video():
    class DuplicateCore(FakeCore):
        def step(self, p1=0, p2=0):
            previous = self.frame_hash
            super().step(p1, p2)
            if self.state % 2:
                self.frame_hash = previous
            return self.frame_hash
        def restore(self, state):
            super().restore(state)
            self.frame_hash = None
    clock = Clock()
    result = benchmark(DuplicateCore(clock), 120, 1000,
                       clock=clock, sleep=clock.sleep)
    assert result["gates"]["same_core_replay"]


def test_progress_reports_bounded_monotonic_frame_counts():
    clock = Clock()
    progress = []
    benchmark(FakeCore(clock), 1250, 1, clock=clock, sleep=clock.sleep,
              progress=progress.append)
    assert [row["frames"] for row in progress] == [600, 1200, 1250]
    assert progress[-1]["percent"] == 100
    assert progress[-1]["eta_seconds"] == 0
