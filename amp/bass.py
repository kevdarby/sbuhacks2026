"""A synthesized bass line, played out the Katana's speaker through the
same USB output channels the looper uses (see amp/audio.py).

Notes are scheduled at perf_counter times and rendered inside the audio
callback, which knows (from the stream's own timestamps) when each block
will actually reach the speaker - so a note sounds at its scheduled time
despite the output path's latency, as long as it was scheduled far enough
ahead (SCHEDULE_AHEAD).
"""

import collections
import threading
import time

import numpy as np

# Sound leaves the amp's speaker this long after the time PortAudio reports
# for the block (outputBufferDacTime) - measured with the MacBook mic from
# scheduled 2kHz beeps: +35-36ms in both high- and low-latency stream modes
OUTPUT_EXTRA_DELAY = 0.035
# How far ahead a note must be scheduled so its start isn't already
# rendered. Assumes the stream is opened with latency='low' (7ms blocks,
# ~16ms output latency, plus the delay above); with PortAudio's default
# high latency it would need ~0.4s.
SCHEDULE_AHEAD = 0.12
# A note closer than this to sounding isn't scheduled at all: its start
# would already be rendered (output latency + one block + the delay above)
MIN_BASS_AHEAD = 0.065
# With a low-latency stream, audio reaches the analyzers this soon after
# it's played
AUDIO_DELAY = 0.03
# The chord for a note is read from this much of the latest audio: short,
# so a chord change takes over quickly. Basic Pitch reads notes reliably
# from much less audio than the spectrum (whose low-note resolution needs
# the longer window).
LIVE_WINDOW = 0.3
LIVE_WINDOW_NOTES = 0.15
# Progression memory: cycle lengths tried (beats), how well the last cycle
# must repeat the one before for predictions to be trusted, and how much
# history to keep
CYCLE_BEATS = (4, 8, 16, 32)
CYCLE_MATCH = 0.85
HISTORY_BEATS = 96
# Root notes land in E2-D#3 (82-156Hz). A real bass sits an octave lower,
# but a guitar speaker barely reproduces that; this keeps the fundamental
# audible and the harmonics carry it.
LOWEST_MIDI = 40  # E2
# Level before the amp's playback gain, raised step by step until the bass
# could be heard over the live guitar. pluck() notes peak at 1.0, so 1.5
# matches the loudness (RMS) settled on with the earlier sawtooth sound. A
# note's peak is then ~0.21 at the -17dB tuned gain - ~13dB below clipping.
LEVEL = 1.5
# Plucked-string model (see pluck()). Decay rates are per second: the
# fundamental rings for seconds, each overtone fades faster than the last.
BASE_DECAY = 1.0
OVERTONE_DECAY = 0.35
PLUCK_POSITION = 0.22      # fraction of the string from the bridge
INHARMONICITY = 0.0002
PITCH_DRIFT = 0.004        # +7 cents at the pluck
PLUCK_NOISE = 0.02         # was 0.06 - the click made notes sound choppy
SATURATION = 1.2
MAX_PARTIALS = 40
PARTIAL_CEILING = 4000.0   # Hz - a bass speaker's range, roughly
ATTACK = 0.006
# Long enough to overlap the next note's start, so notes connect
RELEASE = 0.06


def root_midi(pitch_class, octave=0):
    """MIDI note for a root. octave=-1 is a real bass's range (E1-D#2,
    41-78Hz): the speaker loses most of the fundamental there, but the
    synth's harmonics carry the pitch."""
    return LOWEST_MIDI + (pitch_class - LOWEST_MIDI) % 12 + 12 * octave


def midi_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def root_freq(pitch_class, octave=0):
    return midi_freq(root_midi(pitch_class, octave))


# How the bass line moves:
#   drive    eighth notes on the root, an octave jump on the "&" of beat 2,
#            and a half-step approach note into each chord change (once the
#            progression is learned, so the change is known in advance) -
#            punk/pop-punk style, e.g. Green Day
#   eighths  plain eighth notes on the root
#   kick     the root wherever the drum groove hits a beat or accent
#   pedal    one note held under every chord (eighths, octave jumps on the
#            "&" of 2 and 4) - the note the guitar sounds most often, or one
#            given. Needs no chord detection, so it's never wrong or late:
#            the most common note is by definition in most of the chords
BASS_STYLES = ("drive", "eighths", "kick", "pedal")
# The pedal note is the guitar's most frequent pitch class over this many
# beats, and a new note has to beat the current one by PEDAL_MARGIN to take
# over, so it doesn't wander from section to section
PEDAL_BEATS = 48
PEDAL_MARGIN = 1.15
# Auto level: notes are scaled by the guitar's recent loudness relative to
# REF_RMS (a typical strumming level in the recorded takes, 0.017-0.025),
# kept within these bounds so a pause or a huge strum doesn't silence or
# blast the bass. AUTO_TRIM sets the overall balance against the guitar:
# at full LEVEL the bass drowned out a typical strum, and at -6dB and
# -12dB it was still too loud.
LEVEL_SECONDS = 1.5
REF_RMS = 0.022
LEVEL_RANGE = (0.3, 1.8)
AUTO_TRIM = 0.125          # -18dB
# Accents within the bar: on the beat full, off-beat eighths softer, as a
# bass player picking steady eighths does
OFFBEAT_VELOCITY = 0.65


class BassLine:
    """Decides the bass notes from the chord the guitar is playing, in one
    of BASS_STYLES. Called by BeatFollower for steps SCHEDULE_AHEAD in the
    future.

    A note on beat 1 of a new chord has to be chosen before the guitar has
    played that chord, so following what's heard is always late on chord
    changes. So the chord on every beat is remembered, and once the
    progression is heard repeating (one cycle matching the one before),
    the bass plays the chord the cycle says comes next - landing changes on
    the beat. Playing something new breaks the match and it goes back to
    following."""

    def __init__(self, synth, chords, octave=0, style="drive", pedal_note=None):
        """pedal_note: pitch class for the 'pedal' style, or None to use the
        guitar's most common note."""
        self.synth = synth
        self.chords = chords
        self.octave = octave
        self.style = style
        self.pedal_note = pedal_note
        self._pedal = pedal_note
        self._beat_chroma = collections.deque(maxlen=PEDAL_BEATS)
        self.chord = None       # (root, quality) last used
        self.cycle = None       # progression length in beats, while predicting
        self._beat_times = {}   # beat index -> start time
        self._history = {}      # beat index -> (root, quality) or None

    def reset(self):
        """Forget the progression - when the beat grid is renumbered."""
        self.cycle = None
        self._beat_times = {}
        self._history = {}

    def beat(self, index, start, beat_seconds):
        """Called once per beat by the follower with the grid; records the
        chord of every beat whose audio has fully arrived."""
        self._beat_times[index] = start
        latest = time.perf_counter() - AUDIO_DELAY
        for j in sorted(self._beat_times):
            if j in self._history:
                continue
            t = self._beat_times[j]
            if t + beat_seconds > latest:
                break
            # Skip the very start of the beat: the strum's attack is noise
            self._history[j] = self.chords.chord_between(t + 0.1 * beat_seconds, t + beat_seconds,
                                                         sticky=False)
            if self.style == "pedal":
                profile = self.chords.chroma_between(t, t + beat_seconds)
                if profile is not None:
                    self._beat_chroma.append(profile)
        for j in [j for j in self._beat_times if j < index - HISTORY_BEATS]:
            self._beat_times.pop(j)
            self._history.pop(j, None)

    def _predict(self, index):
        """Chord the learned progression puts on beat `index`, or None."""
        known = [j for j in self._history if self._history[j] is not None]
        if not known:
            self.cycle = None
            return None
        last = max(self._history)
        best = None
        for length in CYCLE_BEATS:
            # Did the most recent cycle repeat the one before it?
            pairs = [(self._history.get(j), self._history.get(j - length))
                     for j in range(last - length + 1, last + 1)]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if len(pairs) < 0.75 * length:
                continue
            match = sum(a[0] == b[0] for a, b in pairs) / len(pairs)
            if match >= CYCLE_MATCH and (best is None or match > best[1]):
                best = (length, match)
        if best is None:
            self.cycle = None
            return None
        self.cycle = best[0]
        # Same place in the previous cycle (or the one before, if the
        # previous one's beat hasn't been analyzed yet)
        for back in (self.cycle, 2 * self.cycle):
            chord = self._history.get(index - back)
            if chord is not None:
                return chord
        return None

    def _pedal_note(self):
        """Current pedal pitch class, or None until enough has been heard."""
        if self.pedal_note is not None or len(self._beat_chroma) < 4:
            return self._pedal
        totals = np.sum(self._beat_chroma, axis=0)
        best = int(np.argmax(totals))
        if self._pedal is None or totals[best] > PEDAL_MARGIN * totals[self._pedal]:
            self._pedal = best
        return self._pedal

    def _velocity(self, pos):
        """Loudness for a note at step `pos`: the guitar's current level
        times the accent for that place in the beat."""
        level = self.chords.level(LEVEL_SECONDS) / REF_RMS
        level = min(max(level, LEVEL_RANGE[0]), LEVEL_RANGE[1])
        accent = 1.0 if pos % 4 == 0 else OFFBEAT_VELOCITY
        return AUTO_TRIM * level * accent

    def plan(self, start, duration, beat_seconds, beat_index, pos):
        """Schedule the note for 16th-note step `pos` (0-15) of the bar,
        which falls in beat `beat_index`."""
        if self.style == "pedal":
            pedal = self._pedal_note()
            if pedal is not None:
                self.chord = (pedal, "")
                note = root_midi(pedal, self.octave) + (12 if pos in (6, 14) else 0)
                self.synth.schedule(start, duration, midi_freq(note), self._velocity(pos))
                return
            # Nothing heard yet to pick a note from: follow the chords
        predicted = self._predict(beat_index)
        if predicted is not None:
            self.chord = predicted
        else:
            latest = time.perf_counter() - AUDIO_DELAY
            window = LIVE_WINDOW_NOTES if self.chords.note_model is not None else LIVE_WINDOW
            self.chord = self.chords.chord_between(latest - window, latest)
        if self.chord is None:
            return
        note = root_midi(self.chord[0], self.octave)
        if self.style == "drive" and pos % 4 == 2:
            # The "&" of a beat. Once the progression is known: lead into
            # the next chord if it changes on the next beat, or bounce up an
            # octave if the chord only started this beat - so a two-beat
            # chord goes root, octave, root, approach. Before that, just
            # the octave on the "&" of 2.
            if self.cycle:
                upcoming = self._predict(beat_index + 1)
                previous = self._predict(beat_index - 1)
                if upcoming is not None and upcoming[0] != self.chord[0]:
                    note = self._approach(note, root_midi(upcoming[0], self.octave))
                elif previous is not None and previous[0] != self.chord[0]:
                    note += 12
            elif pos == 6:
                note += 12
        self.synth.schedule(start, duration, midi_freq(note), self._velocity(pos))

    @staticmethod
    def _approach(current, target):
        """A half step into `target` from the side `current` is on, so the
        line slides into the new root (G into Ab from below, E into Eb
        from above)."""
        return target - 1 if current < target else target + 1

    def plays_on(self, symbol, pos):
        """Whether the bass plays at 16th step `pos`, given the drum
        groove's symbol there. 'kick' joins every drum hit on a beat plus
        the groove's off-beat accents (not ghost notes between beats, which
        would make it bounce); the others play straight eighth notes."""
        if self.style == "kick":
            return symbol in ("X", "x") or (symbol == "g" and pos % 2 == 0)
        return pos % 2 == 0


class BassSynth:
    def __init__(self, sample_rate, level_db=0.0):
        self.sample_rate = sample_rate
        self.level = LEVEL * 10 ** (level_db / 20)
        self.lock = threading.Lock()
        self._notes = []  # (start, end, samples)
        self._cache = {}
        # Live volume on top of everything else (e.g. set by a hand gesture):
        # applied as the audio is rendered so it takes effect at once, and
        # ramped across each block so moving it doesn't crackle
        self.master = 1.0
        self._applied_master = 1.0
        # Called with (start, velocity) for each note scheduled, e.g. to
        # draw it - from whichever thread scheduled it
        self.on_note = None

    def schedule(self, start, duration, freq, velocity=1.0):
        """Queue a note at `velocity` times the synth's level. The waveform
        is built here, on the caller's thread, so the audio callback only
        has to copy samples."""
        key = (round(freq, 2), round(duration, 2))
        samples = self._cache.get(key)
        if samples is None:
            samples = self.level * pluck(freq, duration, self.sample_rate)
            if len(self._cache) < 256:
                self._cache[key] = samples
        self.queue(start, velocity * samples)
        if self.on_note is not None:
            self.on_note(start, velocity)

    def queue(self, start, samples):
        """Play ready-made `samples` from perf_counter time `start`."""
        with self.lock:
            self._notes.append((start, start + len(samples) / self.sample_rate, samples))

    def clear(self):
        with self.lock:
            self._notes = []

    def render(self, frames, time_info):
        """Mono samples for one output block, or None if nothing sounds in
        it. Called from the audio callback."""
        # The stream reports times on its own clock; map the block's start
        # onto perf_counter via the callback's current time
        block_start = (time_info.outputBufferDacTime + (time.perf_counter() - time_info.currentTime)
                       + OUTPUT_EXTRA_DELAY)
        block_end = block_start + frames / self.sample_rate
        with self.lock:
            self._notes = [n for n in self._notes if n[1] > block_start]
            sounding = [n for n in self._notes if n[0] < block_end]
        target = self.master
        ramp = np.linspace(self._applied_master, target, frames, dtype=np.float32)
        self._applied_master = target
        if not sounding:
            return None
        out = np.zeros(frames, dtype=np.float32)
        for start, end, samples in sounding:
            # Where this block falls within the note, in samples
            offset = int(round((block_start - start) * self.sample_rate))
            lo, hi = max(offset, 0), min(offset + frames, len(samples))
            if hi > lo:
                out[lo - offset:hi - offset] += samples[lo:hi]
        return out * ramp


def pluck(freq, duration, sample_rate):
    """One plucked bass note, peak 1.0, lasting `duration` plus a short
    release. Modeled on how a real string behaves, since a sawtooth whose
    overtones all faded together (the first version) sounded synthetic -
    the mic showed every harmonic still at full level 1.3s into a note:

    - high overtones die away much faster than low ones, so the note goes
      from a bright thump to a round hum
    - the pluck point (near the bridge) shapes which overtones are strong
    - a stiff string's overtones run slightly sharp of exact multiples
    - the pitch starts a touch sharp and settles as the string relaxes
    - a short noise burst from the finger at the start
    - the note is damped by the fretting hand when it ends, not cut
    """
    n = int((duration + RELEASE) * sample_rate)
    t = np.arange(n) / sample_rate
    # Pitch settling: +PITCH_DRIFT at the pluck, gone in ~40ms. Integrated
    # into the phase so every overtone follows it.
    settle = 0.04
    phase_time = t + PITCH_DRIFT * settle * (1 - np.exp(-t / settle))
    out = np.zeros(n)
    top = min(MAX_PARTIALS, int(PARTIAL_CEILING / freq))
    for k in range(1, top + 1):
        fk = k * freq * np.sqrt(1 + INHARMONICITY * k * k)
        amp = abs(np.sin(np.pi * k * PLUCK_POSITION)) / k ** 0.9
        decay = BASE_DECAY + OVERTONE_DECAY * (k - 1) ** 1.2
        out += amp * np.exp(-decay * t) * np.sin(2 * np.pi * fk * phase_time)
    # Finger noise: a few ms of low-passed noise at the attack
    rng = np.random.default_rng(int(freq * 100))
    noise = rng.standard_normal(int(0.03 * sample_rate))
    noise = np.convolve(noise, np.ones(12) / 12, mode="same") * np.exp(-np.arange(len(noise)) / (0.006 * sample_rate))
    out[:len(noise)] += PLUCK_NOISE * noise * np.abs(out).max()
    # Attack ramp, and damping over RELEASE once the note's time is up
    out *= np.clip(t / ATTACK, 0, 1)
    out *= np.clip(1 - (t - duration) / RELEASE, 0, 1)
    # A little saturation for the growl of a real bass amp
    out = np.tanh(SATURATION * out / np.abs(out).max()) / np.tanh(SATURATION)
    return out.astype(np.float32)
