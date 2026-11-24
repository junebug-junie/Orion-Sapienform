"""The look's rules: when Orion looks at a game screen, and what the page may say about it.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4,
sections 3 ("The look"), 6 part 8 (Orion's own switch) and "Metric quality gate records: Screen change".
Pure: no I/O, no clock of its own, no numpy. Hub's Retro gateway (services/orion-hub/scripts/retro_watch.py)
is the I/O around it; ``orion/dig/watch.py`` uses the mode rules for Orion's side of the switch.

A look needs something to see. Every 2 s while a sitting is open the gateway reads the newest relayed
frame and measures the screen change (orion/retro/frames.py). Then, in order:

1. The game has no calibrated floor: no look (the page says "no looks for this game yet").
2. The change since the last look is under the game's floor: ``static_skips`` (and ``under_floor_with_input``
   when a button changed or was held in the window: a likely miss).
3. Over the floor (``checks_over_floor``) but the last button change is 60 s old or more: ``input_stale_skips``.
4. Less than 60 s since the last look: wait (the clock is only a ceiling).
5. The watch mode in force is off: no bid.
6. Otherwise one bid in family ``hub.retro_watch``. Only ``funded`` runs the look.

The floor is per game and is set from live play, never by hand (``calibrate``): every ring check of every
game lands in a per-game histogram of its change over the last 30 s (the frame against the check ~30 s
earlier: the look window's own statistic, which separated still from play in step 0, where the 2-second
step did not), split by whether a button changed or was held in those 30 s. A game gets a floor when it has
at least 30 minutes of checks and one value has at most 10% of the no-input checks at or above it and at
least 90% of the input checks at or above it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

FAMILY = "hub.retro_watch"
ACTOR = FAMILY
VERB = "look"
TIER = 1
BINDING_PRODUCER = "orion-hub.retro_watch.binding"
CHOICE_ACTOR = "orion.watch_choice"          # Orion's own switch (dig watch), tier 0, never bids
POOL_ROUTE = "retro_watch"                   # config/gpu_pool.yaml: class chat (the sighted lane), system
POOL_CLASS = "chat"
OBJECT_PREFIX = "retro_sitting:"
GAME_PREFIX = "retro_game:"

CHECK_SEC = 2.0             # one ring check (and one kept packet) every 2 s
RING_MAX = 30               # the last minute of play, still compressed
LOOK_GAP_SEC = 60.0         # at most one look a minute (a ceiling, never a timer)
INPUT_FRESH_SEC = 60.0      # the last button change must be younger than this
FRAMES_PER_LOOK = 4
LOOK_WINDOW_SEC = 30.0      # the frames of a look span the last 30 s at lease grant
MIN_FRAMES = 2              # fewer frames than this: no model call (the prompt states the real count)
WATCHING_FRESH_SEC = 180.0  # "Orion is watching" only while the newest stored note is younger than this
NOTE_MAX_CHARS = 300
MAX_TOKENS = 160
FRAME_TTL_SEC = 3600.0      # the percept store keeps a frame one hour after its LAST post

# Calibration (rev 4 section 3).
CALIBRATION_MIN_SEC = 1800.0
CALIBRATION_WINDOW_SEC = 30.0       # each check's histogram value: its change against the check this long ago
CALIBRATION_MIN_PER_CLASS = 30      # a minute of each kind of check, so neither share is a ratio of 3
NOINPUT_ABOVE_MAX = 0.10
INPUT_ABOVE_MIN = 0.90

# Histogram: bin 0 holds changes under 1e-4; bin k (1..80) holds [EDGES[k-1], EDGES[k]); bin 81 holds 1.0.
# 20 bins a decade: 0.0032 and 0.0047 (Zelda NES walking) fall in different bins, so a floor between them
# is expressible.
BIN_LOW = 1e-4
BINS_PER_DECADE = 20
N_EDGES = 81                                      # 1e-4 .. 1.0
EDGES = tuple(BIN_LOW * 10 ** (k / BINS_PER_DECADE) for k in range(N_EDGES))
N_BINS = N_EDGES + 1

HEAT_PREFIX = "heat:"        # a refusal the heat reflex caused (the gateway tags it; the page names it)
COUNTERS = ("ring_checks", "checks_over_floor", "input_stale_skips", "under_floor_with_input", "static_skips",
            "bids_funded", "model_calls", "notes_stored", "failed_looks")
OUTCOMES = ("stored", "empty", "unclear", "copied", "failed")
VERDICTS = ("unchecked", "checked_holds", "contradicted")

MODES = ("off", "quiet", "on")
MODE_RANK = {m: i for i, m in enumerate(MODES)}
SIDES = ("orion", "juniper")
SCOPES = ("sitting", "game", "all")


def bin_of(value: float) -> int:
    """The histogram bin of one change value (0..1)."""
    if not math.isfinite(value) or value < BIN_LOW:
        return 0
    if value >= 1.0:
        return N_BINS - 1
    k = int(math.floor(BINS_PER_DECADE * math.log10(value / BIN_LOW) + 1e-9)) + 1
    return max(1, min(N_BINS - 1, k))


def share_at_or_above(hist: Sequence[int], edge_index: int) -> float:
    """Share of checks whose value is >= EDGES[edge_index] (bins edge_index+1 and up)."""
    total = sum(hist)
    if total <= 0:
        return 0.0
    return sum(hist[edge_index + 1:]) / total


@dataclass(frozen=True)
class Calibration:
    floor: Optional[float]
    status: str                  # set | too_little_play | one_sided | no_separation
    checks_input: int
    checks_no_input: int
    seconds: float
    noinput_above: Optional[float] = None
    input_above: Optional[float] = None

    def words(self) -> str:
        minutes = int(self.seconds // 60)
        if self.status == "set":
            return (f"floor {self.floor:.4f}: {self.noinput_above:.0%} of still checks and "
                    f"{self.input_above:.0%} of played checks at or above it ({minutes} min of play)")
        if self.status == "too_little_play":
            return f"{minutes} of {int(CALIBRATION_MIN_SEC // 60)} minutes of play measured"
        if self.status == "one_sided":
            return f"{minutes} min measured, but not enough of both still and played checks"
        return f"{minutes} min measured; no screen-change value separates play from stillness in this game"


def _hist(h: Sequence[int]) -> list[int]:
    out = [max(0, int(x)) for x in list(h)[:N_BINS]]
    return out + [0] * (N_BINS - len(out))


def calibrate(hist_input: Sequence[int], hist_no_input: Sequence[int], check_sec: float = CHECK_SEC) -> Calibration:
    """The game's floor from its histograms, or None with the reason. Among the edges that pass the rule,
    the one with the widest gap (input share minus no-input share); ties go to the lower edge, which looks
    more often."""
    hi, hn = _hist(hist_input), _hist(hist_no_input)
    ni, nn = sum(hi), sum(hn)
    seconds = (ni + nn) * float(check_sec)
    if seconds < CALIBRATION_MIN_SEC:
        return Calibration(None, "too_little_play", ni, nn, seconds)
    if ni < CALIBRATION_MIN_PER_CLASS or nn < CALIBRATION_MIN_PER_CLASS:
        return Calibration(None, "one_sided", ni, nn, seconds)
    best = separating_edge(hi, hn)
    if best is None:
        return Calibration(None, "no_separation", ni, nn, seconds)
    k, a_no, a_in = best
    return Calibration(EDGES[k], "set", ni, nn, seconds, round(a_no, 4), round(a_in, 4))


def separating_edge(hist_input: Sequence[int], hist_no_input: Sequence[int]) -> Optional[tuple[int, float, float]]:
    """The rule alone, with no minimum of play: (edge index, no-input share, input share) of the edge with the
    widest gap that passes <= 10% / >= 90%, ties to the lower edge; None when no edge passes. ``calibrate``
    adds the 30-minute minimum; the step 0 eval calls this on the fixture's few sequences."""
    hi, hn = _hist(hist_input), _hist(hist_no_input)
    if sum(hi) <= 0 or sum(hn) <= 0:
        return None
    best: Optional[tuple[float, int, float, float]] = None
    for k in range(N_EDGES):
        a_in, a_no = share_at_or_above(hi, k), share_at_or_above(hn, k)
        if a_no <= NOINPUT_ABOVE_MAX and a_in >= INPUT_ABOVE_MIN:
            gap = a_in - a_no
            if best is None or gap > best[0] + 1e-12:
                best = (gap, k, a_no, a_in)
    return None if best is None else best[1:]


# --- the gate --------------------------------------------------------------------------------------

GATE_NO_FLOOR = "no_floor"
GATE_STATIC = "static"
GATE_INPUT_STALE = "input_stale"
GATE_TOO_SOON = "too_soon"
GATE_READY = "ready"


def gate(change: Optional[float], floor: Optional[float], *, now: float, last_input_at: Optional[float],
         last_look_at: Optional[float], look_gap_sec: float = LOOK_GAP_SEC) -> str:
    """Which gate one ring check stops at (module docstring, steps 1-4)."""
    if floor is None:
        return GATE_NO_FLOOR
    if change is None or change < floor:
        return GATE_STATIC
    if last_input_at is None or now - last_input_at >= INPUT_FRESH_SEC:
        return GATE_INPUT_STALE
    if last_look_at is not None and now - last_look_at < look_gap_sec:
        return GATE_TOO_SOON
    return GATE_READY


def count(counters: dict, gate_result: str, *, input_in_window: bool) -> None:
    """The sitting's counters for one ring check that had a floor to compare with."""
    if gate_result == GATE_NO_FLOOR:
        return
    if gate_result == GATE_STATIC:
        counters["static_skips"] = counters.get("static_skips", 0) + 1
        if input_in_window:
            counters["under_floor_with_input"] = counters.get("under_floor_with_input", 0) + 1
        return
    counters["checks_over_floor"] = counters.get("checks_over_floor", 0) + 1
    if gate_result == GATE_INPUT_STALE:
        counters["input_stale_skips"] = counters.get("input_stale_skips", 0) + 1


def pick_frames(times: Sequence[float], now: float, n: int = FRAMES_PER_LOOK,
                window_sec: float = LOOK_WINDOW_SEC) -> list[int]:
    """Indices of up to ``n`` ring entries spread over the last ``window_sec`` (oldest first, newest last).
    ``times`` is ascending."""
    inside = [i for i, t in enumerate(times) if now - t <= window_sec]
    if len(inside) <= n:
        return inside
    if n == 1:
        return [inside[-1]]
    step = (len(inside) - 1) / (n - 1)
    return sorted({inside[round(j * step)] for j in range(n)})


# --- the note --------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def words(text: Optional[str]) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def similar(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Word overlap of two notes (Jaccard), 0..1; None when either is missing."""
    if not a or not b:
        return None
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 4)


def clean_note(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())[:NOTE_MAX_CHARS]


def classify(note: Optional[str], previous: Optional[str]) -> str:
    """stored | empty | unclear | copied. ``previous``: the sitting's previous note (any outcome)."""
    text = clean_note(note)
    if not text:
        return "empty"
    if _WORD.findall(text.lower()) == ["unclear"]:
        return "unclear"
    if previous is not None and text == clean_note(previous):
        return "copied"
    return "stored"


def prompt(game_title: Optional[str], minutes_played: float, n_frames: int) -> str:
    """The look's whole prompt: stateless (no earlier note: step 0 saw a model copy one word for word)."""
    title = (game_title or "").strip() or "a game"
    minutes = max(0, int(round(minutes_played)))
    plural = "frame" if n_frames == 1 else "frames"
    return (
        "You are the eyes of Orion for a video game that someone else is playing. You describe. You do not judge, "
        f"and you do not speak to anyone. The game is {title}. It has been played for {minutes} minutes. "
        f"You get {n_frames} {plural} from the last 30 seconds, oldest first. Write one short note of at most "
        f"{NOTE_MAX_CHARS} characters: where the player's character is, what they are doing, and what changed "
        "across the frames. Read any words shown on the screen exactly as written. Do not quote numbers from the "
        "status bar (hearts, counters, money, scores): they are misread. Use only what is visible in the frames. "
        "Do not use a name for the player. If you cannot tell what is shown, write exactly: unclear"
    )


# --- the switch (lower wins) -----------------------------------------------------------------------

@dataclass(frozen=True)
class Mode:
    mode: str                    # off | quiet | on
    set_by: Optional[str]        # orion | juniper | None (both on)
    why: Optional[str] = None


def valid_mode(mode: str) -> str:
    m = str(mode or "").strip().lower()
    if m not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return m


def effective(orion: Optional[str], juniper: Optional[str], *, orion_why: Optional[str] = None,
              juniper_why: Optional[str] = None) -> Mode:
    """The mode in force: the lower of the two sides (unset = on). On a tie below ``on`` Orion's own choice
    is named, so the page never words Orion's decline as Juniper's."""
    o = orion if orion in MODE_RANK else "on"
    j = juniper if juniper in MODE_RANK else "on"
    if MODE_RANK[o] == MODE_RANK[j]:
        if o == "on":
            return Mode("on", None)
        return Mode(o, "orion", orion_why)
    if MODE_RANK[o] < MODE_RANK[j]:
        return Mode(o, "orion", orion_why)
    return Mode(j, "juniper", juniper_why)


def orion_choice(rows: Sequence[dict], sitting_id: Optional[str], game_id: Optional[str]) -> Optional[dict]:
    """Orion's standing choice for this sitting: a choice made for THIS sitting wins over the game's
    default; within each, the newest. ``rows``: {scope, scope_id, mode, why, set_at} of side orion."""
    def newest(scope: str, scope_id: Optional[str]) -> Optional[dict]:
        if not scope_id:
            return None
        mine = [r for r in rows if r.get("scope") == scope and r.get("scope_id") == scope_id]
        return max(mine, key=lambda r: r.get("set_at")) if mine else None
    return newest("sitting", sitting_id) or newest("game", game_id)


# --- what the page says ----------------------------------------------------------------------------

def _mins(sec: float) -> int:
    return max(1, int(sec // 60))


def status(*, enabled: bool, sitting_open: bool, mode: Mode, calibration: Optional[Calibration],
           now: float, last_stored_at: Optional[float], last_refusal: Optional[tuple[str, float]],
           last_over_floor_at: Optional[float], last_input_at: Optional[float],
           sitting_started_at: Optional[float]) -> dict:
    """One status line for the game page. "Orion is watching" only with a stored note under 3 minutes old;
    otherwise why not. ``last_refusal``: (reason, at) of the newest bid or lease that was refused."""
    def out(code: str, text: str) -> dict:
        return {"code": code, "text": text}

    if not enabled:
        return out("disabled", "Orion's watch is switched off in Hub's settings.")
    if not sitting_open:
        return out("no_sitting", "Nobody is playing.")
    if mode.mode == "off":
        if mode.set_by == "orion":
            return out("off_orion", "Orion chose not to watch" + (f": {mode.why}" if mode.why else "."))
        return out("off_juniper", "The watch is off (the switch on this page).")
    if calibration is None or calibration.floor is None:
        detail = calibration.words() if calibration is not None else "nothing measured yet"
        return out("no_floor", f"No looks for this game yet: {detail}.")
    if last_stored_at is not None and now - last_stored_at < WATCHING_FRESH_SEC:
        return out("watching", "Orion is watching.")
    if last_refusal is not None and (last_stored_at is None or last_refusal[1] > last_stored_at):
        reason = last_refusal[0]
        if reason.startswith(HEAT_PREFIX):        # set by the gateway from the pool's own refusal text
            return out("refused_heat", f"Orion is not looking: the body is too hot ({reason}).")
        return out("refused", f"Orion is not looking: bid refused ({reason}).")
    since_change = last_over_floor_at if last_over_floor_at is not None else sitting_started_at
    if since_change is not None and now - since_change >= LOOK_GAP_SEC:
        return out("static", f"Orion is looking; the screen has not changed for {_mins(now - since_change)} min.")
    if last_input_at is not None and now - last_input_at >= INPUT_FRESH_SEC:
        return out("input_stale", f"Orion is looking; no button press for {_mins(now - last_input_at)} min.")
    return out("waiting", "Orion is looking; no note yet.")


__all__ = [
    "ACTOR", "BINDING_PRODUCER", "CHOICE_ACTOR", "COUNTERS", "Calibration", "EDGES", "FAMILY", "FRAMES_PER_LOOK",
    "MODES", "Mode", "N_BINS", "OUTCOMES", "POOL_CLASS", "POOL_ROUTE", "VERB", "bin_of", "calibrate", "classify",
    "clean_note", "count", "effective", "gate", "orion_choice", "pick_frames", "prompt", "similar", "status",
    "valid_mode",
]
