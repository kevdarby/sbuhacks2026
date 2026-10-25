"""Which chord the guitar is playing, from the same USB input the tuner and
tempo tracker read (see amp/audio.py). Used to pick the bass note.

Chroma (energy per pitch class, C through B) over a stretch of recent
audio, matched against major, minor and power-chord templates. Only the
root really matters for the bass, and the templates are what get the root
right: a raw "loudest pitch class" is often the fifth or an overtone.

The templates include each chord tone's overtones (an E brings a B with
it, a B brings an F#). Plain three-note templates heard open Em as B or
Bm about half the time on a real recorded take: the pickup barely carries
the low E string's fundamental, the shape has B on two strings, and the
amp's distortion adds more B and F#. Telling the detector the song's
chords (allowed=) is far more reliable still - it then only has to tell
those apart.

Given a NoteModel (amp/notes.py, Basic Pitch) and the dry guitar signal,
the pitch-class profile comes from the notes the network hears instead -
it already separates notes from their overtones, so it's matched against
plain three-note templates. Falls back to the spectrum when the dry signal
isn't there.
"""

import threading

import numpy as np

from amp.tuner import NOTE_NAMES

BUFFER_SECONDS = 4.0
FMIN = 70.0     # below the low E string's fundamental
FMAX = 1000.0   # fundamentals plus the lower overtones
SILENCE_RMS = 0.003
# Template weights per interval above the root
TEMPLATES = {
    "": {0: 1.0, 4: 0.8, 7: 0.8},      # major
    "m": {0: 1.0, 3: 0.8, 7: 0.8},     # minor
    "5": {0: 1.0, 7: 0.9},             # power chord
}
# Overtone k of a note lands this many semitones (mod 12) above it, with
# weight w ** OVERTONE_DECAY - k=1..6: root, octave, fifth, 2 octaves,
# major third, fifth. 0.7 scored best on a recorded take.
OVERTONES = [(0, 1.0), (0, 0.5), (7, 0.33), (0, 0.25), (4, 0.2), (7, 0.17)]
OVERTONE_DECAY = 0.7
# A new chord has to beat the current one's match by this much (cosine
# similarity) to replace it, so passing notes and strum noise don't flip the
# bass back and forth
SWITCH_MARGIN = 0.02


def _chord_vector(root, quality):
    v = np.zeros(12)
    for interval, weight in TEMPLATES[quality].items():
        for overtone, w in OVERTONES:
            v[(root + interval + overtone) % 12] += weight * w ** OVERTONE_DECAY
    return v / np.linalg.norm(v)


def _plain_vector(root, quality):
    v = np.zeros(12)
    for interval, weight in TEMPLATES[quality].items():
        v[(root + interval) % 12] += weight
    return v / np.linalg.norm(v)


CHORD_VECTORS = {(r, q): _chord_vector(r, q) for q in TEMPLATES for r in range(12)}
# For profiles of notes actually played (no overtones in them)
PLAIN_VECTORS = {(r, q): _plain_vector(r, q) for q in TEMPLATES for r in range(12)}
# Below this the dry channel is treated as absent and the spectrum is used
DRY_SILENCE_RMS = 0.0003
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def parse_chords(text):
    """'Em G D A C' (or comma-separated) -> set of (root, quality). Extras
    like 7, maj7 or sus are ignored - only the root and major/minor/power
    matter for picking a bass note."""
    out = set()
    for token in text.replace(",", " ").split():
        token = token[0].upper() + token[1:]
        split = 2 if len(token) > 1 and token[1] in "#b" else 1
        name, rest = FLATS.get(token[:split], token[:split]), token[split:]
        if name not in NOTE_NAMES:
            raise ValueError(f"can't read chord {token!r}")
        if rest.startswith("5"):
            quality = "5"
        elif rest.startswith("m") and not rest.startswith("maj"):
            quality = "m"
        else:
            quality = ""
        out.add((NOTE_NAMES.index(name), quality))
    return out


def chroma(audio, sample_rate):
    """12-bin pitch-class energy profile, normalized to sum 1."""
    n = len(audio)
    spec = np.abs(np.fft.rfft(audio * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    keep = (freqs >= FMIN) & (freqs <= FMAX)
    midi = 69 + 12 * np.log2(freqs[keep] / 440.0)
    pitch_class = np.round(midi).astype(int) % 12
    out = np.bincount(pitch_class, weights=spec[keep], minlength=12)
    # Square root so one loud string doesn't drown out the rest of the chord
    out = np.sqrt(out)
    total = out.sum()
    return out / total if total > 0 else out


def chord_score(profile, root, quality, plain=False):
    """Cosine similarity between a pitch-class profile and a chord's
    template - plain for a profile of played notes, else with overtones."""
    norm = np.linalg.norm(profile)
    vectors = PLAIN_VECTORS if plain else CHORD_VECTORS
    return float(profile @ vectors[(root, quality)] / norm) if norm > 0 else 0.0


def match_chord(profile, allowed=None, plain=False):
    """Best (root pitch class, quality, score) for a pitch-class profile,
    among `allowed` (root, quality) pairs if given."""
    candidates = allowed or CHORD_VECTORS.keys()
    root, quality = max(candidates, key=lambda c: chord_score(profile, *c, plain=plain))
    return root, quality, chord_score(profile, root, quality, plain=plain)


def chord_name(root, quality):
    return NOTE_NAMES[root] + quality


class ChordDetector:
    """Fed the raw input blocks alongside the tempo tracker (feed() only
    copies), and asked for the chord over a time range by the bass, which
    runs on its own schedule. Uses the tracker's sample clock so 'the last
    beat' means the same thing to both."""

    def __init__(self, sample_rate, tracker, allowed=None, note_model=None):
        """allowed: the song's chords as (root, quality) pairs (see
        parse_chords), or None to consider every chord. note_model: a
        NoteModel to read chords from the dry signal (fed via feed_dry)."""
        self.sample_rate = sample_rate
        self.tracker = tracker
        self.allowed = allowed
        self.note_model = note_model
        self.lock = threading.Lock()
        self._buffer = np.zeros(int(BUFFER_SECONDS * sample_rate), dtype=np.float32)
        # Dry signal, kept sample-aligned with _buffer: both are fed every
        # block of the same callback
        self._dry = np.zeros_like(self._buffer)
        self._samples_fed = 0
        self.current = None  # (root, quality) or None
        self.source = None   # what the last chord was read from

    def feed(self, stereo_block):
        mono = stereo_block.mean(axis=1).astype(np.float32)
        n = min(len(mono), len(self._buffer))
        with self.lock:
            self._buffer[:-n] = self._buffer[n:]
            self._buffer[-n:] = mono[-n:]
            self._samples_fed += len(mono)

    def feed_dry(self, stereo_block):
        """The dry channels of the same block just passed to feed()."""
        mono = stereo_block.mean(axis=1).astype(np.float32)
        n = min(len(mono), len(self._dry))
        with self.lock:
            self._dry[:-n] = self._dry[n:]
            self._dry[-n:] = mono[-n:]

    def _audio_between(self, start, end, dry=False):
        """Buffered input between two perf_counter times, or None if that
        stretch is too short or no longer buffered."""
        first = self.tracker.time_to_sample(start)
        last = self.tracker.time_to_sample(end)
        if first is None:
            return None
        with self.lock:
            fed = self._samples_fed
            buf_start = fed - len(self._buffer)
            first, last = max(int(first), buf_start), min(int(last), fed)
            if last - first < self.sample_rate * 0.1:
                return None
            source = self._dry if dry else self._buffer
            return source[first - buf_start:last - buf_start].copy()

    def _profile(self, audio, start, end):
        """(pitch-class profile, whether it's of played notes) for a stretch
        of input: from Basic Pitch on the dry signal when possible."""
        if self.note_model is not None:
            dry = self._audio_between(start, end, dry=True)
            if dry is not None and np.sqrt(np.mean(dry ** 2)) >= DRY_SILENCE_RMS:
                profile = self.note_model.pitch_classes(dry, self.sample_rate)
                if profile is not None:
                    self.source = "basic pitch"
                    return profile, True
        self.source = "spectrum"
        return chroma(audio, self.sample_rate), False

    def level(self, seconds):
        """RMS of the most recent `seconds` of input - how hard the guitar is
        being played right now."""
        n = int(seconds * self.sample_rate)
        with self.lock:
            recent = self._buffer[-n:]
            return float(np.sqrt(np.mean(recent ** 2)))

    def chroma_between(self, start, end):
        """Pitch-class profile between two perf_counter times, or None if
        silent or not buffered."""
        audio = self._audio_between(start, end)
        if audio is None or np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS:
            return None
        return chroma(audio, self.sample_rate)

    def chord_between(self, start, end, sticky=True):
        """Chord heard between two perf_counter times, as (root, quality),
        or None if that stretch was silent or isn't buffered. Sticky keeps
        the previous chord unless a different one clearly wins (for live
        bass notes); not sticky just reports the best match (for recording
        what each beat was)."""
        fallback = self.current if sticky else None
        audio = self._audio_between(start, end)
        if audio is None:
            return fallback
        if np.sqrt(np.mean(audio ** 2)) < SILENCE_RMS:
            if sticky:
                self.current = None
            return None
        profile, plain = self._profile(audio, start, end)
        root, quality, score = match_chord(profile, self.allowed, plain)
        if not sticky:
            return root, quality
        # Only the spectrum needs the stickiness: Basic Pitch's notes don't
        # flicker the same way, and holding on cost it 6 points of right
        # roots on a take with quick chord changes
        if not plain and self.current is not None and (root, quality) != self.current:
            if score < chord_score(profile, *self.current, plain=plain) + SWITCH_MARGIN:
                return self.current
        self.current = (root, quality)
        return self.current
