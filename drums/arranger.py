"""Writes a drum groove to fit what the guitar is playing. Reads the
player's rhythm off the tracker's onset envelope (amp/tempo.py) on the
drums' own 16th-note grid, then builds a pattern for the next bar:

- A backbone that fills the space the guitar leaves: half time (beats 1
  and 3) under slow ringing chords, eighth notes under a guitar that only
  strums on the beats, every beat otherwise.
- Extra hits on the guitar's strongest off-beat accents, so the drum locks
  in with the riff's syncopation instead of ignoring it. Busy strumming
  gets fewer of these: doubling every 16th would just be clutter.
- A fill on the last bar of every 8-bar phrase.

All hits sound equally loud on this solenoid (see drums/patterns.py), so
the groove is shaped by where the hits go, not how hard they are.
"""

import math

import numpy as np

from drums.patterns import PULSES
from drums.relay import RELAY_RECOVERY

STEPS = 16
# How many past bars the guitar's pattern is averaged over, newest weighted
# most - enough to smooth out one sloppy bar, few enough to follow a change
# of riff within a couple of bars
PROFILE_BARS = 4
BAR_WEIGHTS = [1.0, 0.8, 0.6, 0.5]
# A strum counts as "played" at a step above this fraction of the bar's
# strongest, and as an accent worth doubling on the drum above ACCENT
ACTIVE = 0.35
ACCENT = 0.6
SPARSE_MAX = 2   # active steps per bar at or below this -> half-time backbone
BUSY_MIN = 10    # at or above this -> fewer accent hits
PHRASE_BARS = 8


def strum_profile(env, times, bar_start, step_seconds, bars=PROFILE_BARS):
    """Average onset strength at each of the 16 steps over the last `bars`
    bars before `bar_start`, scaled to 0..1. None if there isn't at least
    one bar of audio to read."""
    if len(times) == 0:
        return None
    bar_len = STEPS * step_seconds
    window = step_seconds * 0.35
    total = np.zeros(STEPS)
    weight = np.zeros(STEPS)
    for k in range(1, bars + 1):
        start = bar_start - k * bar_len
        if start < times[0]:
            break
        for j in range(STEPS):
            t = start + j * step_seconds
            # The newest bar may not be fully analyzed yet; skip what isn't
            if t + window > times[-1]:
                continue
            lo, hi = np.searchsorted(times, [t - window, t + window])
            if hi > lo:
                total[j] += BAR_WEIGHTS[k - 1] * env[lo:hi].max()
                weight[j] += BAR_WEIGHTS[k - 1]
    if not weight.any():
        return None
    profile = np.where(weight > 0, total / np.maximum(weight, 1e-9), 0)
    lo, hi = profile.min(), profile.max()
    if hi - lo < 1e-9:
        return None
    return (profile - lo) / (hi - lo)


def downbeat(profile):
    """Which beat (0-3) of the drums' current bar the player treats as beat
    1: the most accented one, with the beat two later (beat 3 of 4/4) as
    supporting evidence. Returns (beat, how much it beat the runner-up)."""
    scores = [profile[4 * b] + 0.5 * profile[(4 * b + 8) % STEPS] for b in range(4)]
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    return int(order[0]), best / max(second, 1e-9)


def _min_gap_steps(step_seconds):
    """Closest two hits can be, in steps, for the relay to keep up with the
    shortest pulse used here."""
    return max(1, math.ceil((PULSES["g"] + RELAY_RECOVERY) / step_seconds))


def _fits(pos, chosen, gap):
    return all(min(abs(pos - c), STEPS - abs(pos - c)) >= gap for c in chosen)


def _feel_from_guitar(profile):
    """(backbone steps, max accent hits, description) filling the space the
    guitar leaves rather than copying it: a guitar strumming only on the
    beats gets eighth notes under it to drive it along, a busy one gets a
    plain backbone."""
    played = {j for j in range(STEPS) if profile[j] >= ACTIVE}
    if len(played) <= SPARSE_MAX:
        return [0, 8], 1, "half time"
    if played <= {0, 4, 8, 12}:
        return list(range(0, STEPS, 2)), 0, "driving eighths"
    if len(played) >= BUSY_MIN:
        return [0, 4, 8, 12], 2, "straight, busy guitar"
    return [0, 4, 8, 12], 3, "straight"


# Intensity 0..1 (e.g. from a hand gesture), calm to crazy:
#   (upper bound, backbone, accent hits, fill every N bars or None, ghost-note
#   chance, description). The middle band keeps the guitar-driven feel.
INTENSITY_LEVELS = [
    (0.2, [0, 8], 0, None, 0.0, "calm"),
    (0.4, [0, 4, 8, 12], 1, PHRASE_BARS, 0.0, "steady"),
    (0.6, None, None, PHRASE_BARS, 0.0, None),
    (0.8, list(range(0, STEPS, 2)), 2, 4, 0.0, "driving"),
    (1.01, list(range(0, STEPS, 2)), 3, 2, 0.3, "crazy"),
]


def arrange(profile, step_seconds, bar_index, intensity=None):
    """Pattern string for one bar (same notation as drums/patterns.py) plus
    a short description of what it's doing. intensity (0..1) overrides how
    busy it is; None leaves it to the guitar's feel."""
    fill_every, ghost_chance = PHRASE_BARS, 0.0
    backbone, extras, feel = _feel_from_guitar(profile)
    if intensity is not None:
        level = next(lv for lv in INTENSITY_LEVELS if intensity < lv[0])
        _, level_backbone, level_extras, fill_every, ghost_chance, level_feel = level
        if level_backbone is not None:
            backbone, extras, feel = level_backbone, level_extras, level_feel

    gap = _min_gap_steps(step_seconds)
    chosen = []
    for pos in backbone:
        if _fits(pos, chosen, gap):
            chosen.append(pos)
    # The guitar's strongest accents off the backbone, local peaks only so
    # one long strum doesn't claim two neighbouring steps
    candidates = [j for j in range(STEPS) if j not in backbone
                  and profile[j] >= ACCENT
                  and profile[j] >= profile[(j - 1) % STEPS]
                  and profile[j] >= profile[(j + 1) % STEPS]]
    candidates.sort(key=lambda j: profile[j], reverse=True)
    accents = []
    for pos in candidates:
        if len(accents) >= extras:
            break
        if _fits(pos, chosen, gap):
            chosen.append(pos)
            accents.append(pos)

    if ghost_chance:
        # Scattered extra 16ths wherever the relay can fit them - a
        # different scatter every bar, so it stays restless
        rng = np.random.default_rng(bar_index)
        for pos in rng.permutation(STEPS):
            if pos not in chosen and rng.random() < ghost_chance and _fits(pos, chosen, gap):
                chosen.append(int(pos))

    fill = fill_every is not None and bar_index % fill_every == fill_every - 1
    if fill:
        # Pack the end of the bar as tightly as the relay allows: the last
        # beat, or the last two at full intensity
        for pos in range(8 if ghost_chance else 12, STEPS):
            if pos not in chosen and _fits(pos, chosen, gap):
                chosen.append(pos)

    pattern = ["."] * STEPS
    for pos in chosen:
        pattern[pos] = "x" if pos % 4 == 0 else "g"
    pattern[0] = "X"
    desc = feel
    if accents:
        desc += f", +{len(accents)} accent{'s' * (len(accents) > 1)}"
    if fill:
        desc += ", fill"
    return "".join(pattern), desc


class Arranger:
    """Called by BeatFollower at the start of each bar for the next bar's
    pattern. Also works out where beat 1 is, which the follower can't know
    on its own - it just starts its bar on whichever beat it came in on."""

    def __init__(self, tracker, fallback):
        self.tracker = tracker
        self.fallback = fallback
        self.bar_index = 0
        self.profile = None
        self.pattern = fallback
        self.description = "waiting for a bar of guitar"
        self._downbeat_checked = False
        # 0..1 calm-to-crazy override (see INTENSITY_LEVELS), or None to let
        # the guitar decide; read at the start of each bar
        self.intensity = None

    def reset(self):
        self.bar_index = 0
        self.profile = None
        self.pattern = self.fallback
        self.description = "waiting for a bar of guitar"
        self._downbeat_checked = False

    def next_bar(self, bar_start, step_seconds):
        """Returns (pattern, beats to shift the bar by so beat 1 lines up)."""
        env, times = self.tracker.onset_envelope()
        profile = strum_profile(env, times, bar_start, step_seconds)
        shift = 0
        if profile is not None:
            # Once there are two bars to go on, move the drums' beat 1 onto
            # the player's. Only once: re-deciding it every bar would make
            # the groove jump around whenever the accents are ambiguous.
            if not self._downbeat_checked and bar_start - 2 * STEPS * step_seconds >= times[0]:
                self._downbeat_checked = True
                beat, margin = downbeat(profile)
                if beat and margin > 1.15:
                    shift = beat
                    profile = np.roll(profile, -4 * beat)
            self.profile = profile
            self.pattern, self.description = arrange(profile, step_seconds, self.bar_index,
                                                     self.intensity)
        self.bar_index += 1
        return self.pattern, shift


def chart(values_or_pattern):
    """One-line view of a bar, beat numbers under the downbeats, e.g. for a
    profile: '█ · ▄ · │ █ ...'."""
    cells = []
    for j in range(STEPS):
        v = values_or_pattern[j]
        if isinstance(v, str):
            cells.append(v)
        else:
            cells.append("█" if v >= ACCENT else "▄" if v >= ACTIVE else "·")
        if j % 4 == 3 and j < STEPS - 1:
            cells.append("|")
    return " ".join(cells)
