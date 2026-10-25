"""Software drum, for playing without the solenoid: a stand-in for the relay
that, instead of closing it, schedules a synthesized drum hit out the amp's
speaker - mixed in with the bass, through the same scheduling (see
amp/bass.py). It mimics the solenoid: one drum, and every hit the same,
since on the rig every pulse length was measured hitting equally hard (see
drums/patterns.py).
"""

import time

import numpy as np

from amp.bass import MIN_BASS_AHEAD, BassSynth
from drums.relay import MAX_PULSE

# How far ahead of a hit the follower fires the software relay. Has to be at
# least MIN_BASS_AHEAD, or the hit's attack would already be rendered by
# the time it's scheduled.
SOFTWARE_LEAD = MIN_BASS_AHEAD + 0.02
# Peak level before the amp's playback gain, alongside the bass's 1.5
LEVEL = 1.4


def drum_hit(sample_rate):
    """A plunger striking a drum: the head's thump (a tone that drops in
    pitch as it settles), its ring, and the tap of the plunger."""
    t = np.arange(int(0.3 * sample_rate)) / sample_rate
    freq = 120 + 80 * np.exp(-t / 0.02)
    thump = np.sin(2 * np.pi * np.cumsum(freq) / sample_rate) * np.exp(-t / 0.09)
    ring = 0.25 * np.sin(2 * np.pi * 310 * t) * np.exp(-t / 0.06)
    tap = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.004) * 0.5
    sound = np.tanh(1.3 * (thump + ring + tap))
    sound[:64] *= np.linspace(0, 1, 64)  # so it doesn't start with a click of its own
    return sound


class DrumSynth(BassSynth):
    """BassSynth's scheduling and rendering, playing the drum hit."""

    def __init__(self, sample_rate):
        super().__init__(sample_rate)
        sound = drum_hit(sample_rate)
        self.sound = (LEVEL * sound / np.abs(sound).max()).astype(np.float32)

    def play(self, start):
        self.queue(start, self.sound)


class SoftwareRelay:
    """Takes the relay's place: each hit plays the drum SOFTWARE_LEAD later -
    give the follower lead=SOFTWARE_LEAD so it lands on the beat. Takes the
    same time per hit as the real relay, so the timing behaves the same."""

    port = "software"

    def __init__(self, synth, lead=SOFTWARE_LEAD):
        self.synth = synth
        self.lead = lead

    def hit(self, pulse):
        self.synth.play(time.perf_counter() + self.lead)
        time.sleep(min(pulse, MAX_PULSE))

    def close(self):
        pass
