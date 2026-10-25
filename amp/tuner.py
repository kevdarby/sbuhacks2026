"""Live pitch detection from the guitar signal, via the same USB audio
input the looper reads (see amp/audio.py). Pure signal analysis - doesn't
touch the amp at all, works entirely from what's captured over USB.
"""

import math
import threading

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Guitar range: low E ~82Hz to well above the 12th-fret harmonics. A window
# of ~2048 samples (~46ms @ 44.1kHz) covers a few periods even of the low E
# string, which the difference function needs to lock on reliably.
WINDOW_SIZE = 2048
FMIN = 70.0
FMAX = 1200.0
# YIN threshold on the cumulative mean normalized difference function - the
# first lag whose dip goes below this is accepted as the period. Standard
# YIN papers use ~0.1-0.15.
YIN_THRESHOLD = 0.15
SILENCE_RMS = 0.003


def freq_to_note(freq):
    if freq is None or freq <= 0:
        return None
    midi = 69 + 12 * math.log2(freq / 440.0)
    midi_round = round(midi)
    cents = (midi - midi_round) * 100
    name = NOTE_NAMES[int(midi_round) % 12]
    octave = int(midi_round) // 12 - 1
    return f"{name}{octave}", cents


def detect_pitch(buf, sample_rate, fmin=FMIN, fmax=FMAX, threshold=YIN_THRESHOLD):
    """YIN pitch detection (de Cheveigne & Kawahara). A guitar signal is
    correlated with itself not just at its true period but at every integer
    multiple of it, so plain autocorrelation (tried first here) kept
    locking onto the wrong lag - sometimes a spurious short one, sometimes
    an octave-down multiple - depending on which happened to score highest.
    YIN's difference function plus cumulative-mean normalization, combined
    with taking the *first* dip below a threshold rather than the global
    best, is the standard fix for exactly this failure mode."""
    buf = buf - np.mean(buf)
    rms = float(np.sqrt(np.mean(buf ** 2)))
    if rms < SILENCE_RMS:
        return None

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(len(buf) - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return None

    # Difference function d(tau) for every tau from 1 up to max_lag - the
    # cumulative-mean normalization below needs the full run from tau=1,
    # not just the fmin/fmax-restricted range.
    #
    # Computed via FFT rather than a Python loop over tau. Expanding the
    # square gives d(tau) = sum(x[j]^2) + sum(x[j+tau]^2) - 2*corr(tau),
    # where the two sums are cumulative-sum lookups and corr is the linear
    # autocorrelation - one FFT instead of ~600 iterations. This runs on
    # the video thread every frame, and as a Python loop it held the GIL
    # long enough to be a prime suspect for starving the audio callbacks
    # (see ISSUE.md); vectorizing removes it as a variable.
    w = len(buf)
    power = buf.astype(np.float64) ** 2
    cumpower = np.concatenate([[0.0], np.cumsum(power)])
    taus = np.arange(max_lag + 1)
    # sum of x[j]^2 for j in [0, w-tau)  and  for j in [tau, w)
    head = cumpower[w - taus]
    tail = cumpower[w] - cumpower[taus]

    nfft = 1 << int(np.ceil(np.log2(2 * w)))
    spec = np.fft.rfft(buf, nfft)
    corr = np.fft.irfft(spec * np.conj(spec), nfft)[:max_lag + 1]

    d = head + tail - 2 * corr
    d[0] = 0.0
    np.maximum(d, 0.0, out=d)  # guard tiny negative values from FFT roundoff

    running = np.cumsum(d[1:])
    cmndf = np.ones(max_lag + 1)
    np.divide(d[1:] * taus[1:], running, out=cmndf[1:], where=running > 0)

    candidate = None
    for tau in range(min_lag, max_lag + 1):
        if cmndf[tau] < threshold:
            candidate = tau
            break
    if candidate is None:
        return None

    # Walk forward to the local minimum past the threshold crossing - YIN's
    # dip is a valley, and the first sample under threshold isn't always
    # its lowest point.
    tau = candidate
    while tau + 1 <= max_lag and cmndf[tau + 1] < cmndf[tau]:
        tau += 1

    confidence = max(0.0, min(1.0, 1.0 - cmndf[tau]))
    freq = sample_rate / tau
    return freq, confidence


class PitchAnalyzer:
    """Fed mono-mixed audio blocks from the realtime audio callback (cheap:
    just copies into a rolling buffer); the actual pitch detection runs
    separately (call detect() from the video loop, not the audio thread) so
    the heavier computation never risks glitching audio playback."""

    def __init__(self, sample_rate, window_size=WINDOW_SIZE):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.buffer = np.zeros(window_size, dtype=np.float32)
        self.lock = threading.Lock()

    def feed(self, stereo_block):
        mono = stereo_block.mean(axis=1)
        n = len(mono)
        with self.lock:
            if n >= self.window_size:
                self.buffer[:] = mono[-self.window_size:]
            else:
                self.buffer[:-n] = self.buffer[n:]
                self.buffer[-n:] = mono

    def detect(self):
        with self.lock:
            snapshot = self.buffer.copy()
        result = detect_pitch(snapshot, self.sample_rate)
        if result is None:
            return None
        freq, confidence = result
        note = freq_to_note(freq)
        if note is None:
            return None
        name, cents = note
        return freq, name, cents, confidence
