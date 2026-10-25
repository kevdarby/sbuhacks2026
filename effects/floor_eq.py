"""A spectrum analyser as LED bars rising from the bottom edge of the
picture: lows on the left in the drums' blue, through to highs on the right
in the bass's red. It shows the whole band - your guitar (the dry input)
and the bass and software drums (what goes out the amp) - each scaled to
its own recent peak, so a quiet one isn't lost under a loud one.

The audio callback only copies samples in (feed_guitar, feed_band); the
FFT runs in the video loop, in draw().
"""

import threading

import cv2
import numpy as np

from amp.tuner import SILENCE_RMS
from effects.stage_lights import BASS_COLOR, DRUM_COLOR

FFT_SIZE = 2048      # ~46ms at 44.1kHz
BARS = 40
LOW_HZ, HIGH_HZ = 50.0, 5000.0  # little in guitar or bass above this
RANGE_DB = 30.0      # a bar spans this far below the source's recent peak
CURVE = 1.6          # >1: only the strong parts of the spectrum stand tall
PEAK_FALL_DB = 6.0   # per second: how fast the reference peak relaxes
BAND_SILENCE_RMS = 1e-4
FALL = 2.5           # per second: how fast a bar drops (it rises at once)
# Layout, as fractions of the frame
MAX_HEIGHT = 0.22    # of the height
BAR_WIDTH = 0.45     # of each bar's slot
CORE_WHITE = 0.15    # how far a bar's core is lifted toward white
BLOOM = 0.6
UNDER_DARKEN = 0.7   # how much of the picture behind a bar is blocked out


class _Source:
    """The latest FFT_SIZE samples of one sound, and the peak level it's
    scaled against."""

    def __init__(self, silence_rms):
        self.buffer = np.zeros(FFT_SIZE, np.float32)
        self.silence_rms = silence_rms
        self.peak_db = -120.0

    def push(self, mono):
        n = len(mono)
        if n >= FFT_SIZE:
            self.buffer[:] = mono[-FFT_SIZE:]
        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = mono

    def levels(self, samples, edges, window, dt):
        """0-1 per bar, or zeros if it's silent."""
        if np.sqrt(np.mean(samples ** 2)) < self.silence_rms:
            return np.zeros(len(edges) - 1, np.float32)
        mag = np.abs(np.fft.rfft(samples * window))
        bands = np.array([mag[lo:hi].max() for lo, hi in zip(edges[:-1], edges[1:])])
        db = 20 * np.log10(bands + 1e-9)
        self.peak_db = max(self.peak_db - PEAK_FALL_DB * dt, db.max())
        return np.clip((db - (self.peak_db - RANGE_DB)) / RANGE_DB, 0, 1) ** CURVE


class FloorEQ:
    def __init__(self, sample_rate):
        self.lock = threading.Lock()
        self.guitar = _Source(SILENCE_RMS)
        self.band = _Source(BAND_SILENCE_RMS)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)
        # FFT bin edges of log-spaced bands, each at least one bin wide
        hz = np.geomspace(LOW_HZ, HIGH_HZ, BARS + 1)
        edges = np.round(hz * FFT_SIZE / sample_rate).astype(int)
        for i in range(1, len(edges)):
            edges[i] = max(edges[i], edges[i - 1] + 1)
        self.edges = edges
        self.values = np.zeros(BARS, np.float32)
        t = np.linspace(0, 1, BARS)[:, None]
        colors = np.asarray(DRUM_COLOR, np.float32) * (1 - t) + np.asarray(BASS_COLOR, np.float32) * t
        # Back to full brightness: halfway between blue and red is a muddy
        # dark purple otherwise
        self.colors = colors * (255 / colors.max(axis=1, keepdims=True))

    def feed_guitar(self, stereo_block):
        """From the audio callback: a block of the dry guitar."""
        with self.lock:
            self.guitar.push(stereo_block.mean(axis=1))

    def feed_band(self, mono, frames):
        """From the audio callback: what the bass and drums play out the amp
        this block, or None if they're silent."""
        with self.lock:
            self.band.push(np.zeros(frames, np.float32) if mono is None else mono)

    def draw(self, frame, dt):
        with self.lock:
            guitar, band = self.guitar.buffer.copy(), self.band.buffer.copy()
        target = np.maximum(self.guitar.levels(guitar, self.edges, self.window, dt),
                            self.band.levels(band, self.edges, self.window, dt))
        self.values = np.maximum(target, self.values - FALL * dt)
        if self.values.max() <= 0.01:
            return

        h, w = frame.shape[:2]
        top = int(h * (1 - MAX_HEIGHT - 0.08))  # room above the bars for glow
        roi = frame[top:]
        rh, rw = roi.shape[:2]
        layer = np.zeros((rh, rw, 3), np.uint8)
        shape = np.zeros((rh, rw), np.uint8)  # the bars alone, for the white core
        slot = w / BARS
        thick = max(2, int(slot * BAR_WIDTH))
        for i, (v, c) in enumerate(zip(self.values, self.colors)):
            if v <= 0.01:
                continue
            x = int((i + 0.5) * slot)
            p1, p2 = (x, rh + thick), (x, int(rh - 4 - v * h * MAX_HEIGHT))
            cv2.line(layer, p1, p2, c.tolist(), thick, cv2.LINE_AA)
            cv2.line(shape, p1, p2, 255, thick, cv2.LINE_AA)

        # Darken the picture under the bars so their colour holds on a
        # bright background; then the core (each bar's colour lifted toward
        # white) and a bloom from a blurred quarter-size copy, added as light
        under = cv2.cvtColor(255 - cv2.multiply(shape, UNDER_DARKEN), cv2.COLOR_GRAY2BGR)
        cv2.multiply(roi, under, dst=roi, scale=1 / 255)
        core = cv2.addWeighted(layer, (1 - CORE_WHITE) * 0.85, layer, 0, 0)
        lift = cv2.cvtColor(cv2.multiply(shape, CORE_WHITE * 0.85), cv2.COLOR_GRAY2BGR)
        small = cv2.resize(layer, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5 * h / 1080),
                           (rw, rh), interpolation=cv2.INTER_LINEAR)
        cv2.add(roi, core, dst=roi)
        cv2.add(roi, lift, dst=roi)
        cv2.addWeighted(roi, 1.0, bloom, BLOOM, 0, dst=roi)
