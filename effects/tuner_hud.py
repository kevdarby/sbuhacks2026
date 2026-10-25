"""The tuner as light, to sit with the LED lighting: the note name glowing
at the top centre, over a long, faint line of light with a bright spot that
slides along it - left of centre when flat, right when sharp. The colour
says how close you are (red, amber, green), and in tune the whole line
lights up green. No box: a soft shadow under the note keeps it readable.
It fades in while a note rings and out shortly after it stops.
"""

import time
from functools import lru_cache

import cv2
import numpy as np

from paths import PROJECT_ROOT

IN_TUNE_CENTS = 5
CLOSE_CENTS = 20
RANGE_CENTS = 50  # the line's ends

IN_TUNE_COLOR = (120, 255, 90)   # BGR neon green
CLOSE_COLOR = (0, 190, 255)      # amber
OFF_COLOR = (60, 60, 255)        # red

# Layout, as fractions of the frame's height (h) or width (w)
NOTE_SIZE = 0.075   # h
NOTE_Y = 0.08       # h, the note's centre
LINE_Y = 0.15       # h
LINE_HALF = 0.2     # w, half the line's length
SPOT_HALF = 0.026   # h, half the spot's length
# How bright the line is when out of tune and in tune
LINE_DIM = 0.25
LINE_LIT = 0.8
# How long the last note stays up after the pitch is lost, and fade speeds
# (per second)
HOLD_SECONDS = 0.6
FADE_IN = 10.0
FADE_OUT = 3.0

FONT_FILE = PROJECT_ROOT / "assets" / "fonts" / "IBMPlexMono-Medium.ttf"


def cents_color(cents):
    mag = abs(cents)
    if mag <= IN_TUNE_CENTS:
        return IN_TUNE_COLOR
    if mag <= CLOSE_CENTS:
        return CLOSE_COLOR
    return OFF_COLOR


@lru_cache(maxsize=64)
def _text_mask(text, size):
    """`text` as a 0-1 float mask, cropped tight, in the project's mono
    font if it's there, OpenCV's own if not."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(FONT_FILE), size)
        left, top, right, bottom = font.getbbox(text)
        img = Image.new("L", (right - left + 8, bottom - top + 8), 0)
        ImageDraw.Draw(img).text((4 - left, 4 - top), text, font=font, fill=255)
        mask = np.asarray(img)
    except Exception:
        scale = size / 30
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        mask = np.zeros((th + base + 8, tw + 8), np.uint8)
        cv2.putText(mask, text, (4, th + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 2, cv2.LINE_AA)
    return mask.astype(np.float32) / 255


def _soften(mask, sigma):
    """`mask` blurred by `sigma` px - at quarter size, since it's all blur
    anyway and full size cost ~20ms a frame."""
    h, w = mask.shape
    small = cv2.resize(mask, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / 4))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


class _Light:
    """Light shapes in one colour, gathered as single-channel amounts of
    colour and of white (cheap), and only coloured in at the end."""

    def __init__(self, shape):
        self.color = np.zeros(shape, np.float32)
        self.white = np.zeros(shape, np.float32)

    def add(self, mask, blur, bloom_gain, core_white):
        """`mask` as light: a core in the colour lifted toward white, plus
        a blurred bloom in the colour."""
        bloom = np.minimum(1, _soften(mask, blur) * bloom_gain)
        self.color += mask * (1 - core_white) + bloom * 0.8
        self.white += mask * core_white

    def bgr(self, color):
        c = cv2.merge([self.color * v for v in color])
        return c + cv2.merge([self.white * 255] * 3)


class TunerDisplay:
    """Owns smoothing state so the reading doesn't jitter frame to frame -
    snaps to a new reading immediately when the note changes, smooths
    within a held note - and the fade in and out."""

    def __init__(self, smoothing=0.25):
        self.smoothing = smoothing
        self.smoothed_cents = None
        self.current_name = None
        self.visible = 0.0
        self._last_heard = -999.0
        self._last_time = time.monotonic()

    def update(self, pitch_result):
        if pitch_result is None:
            return None
        _freq, name, cents, _confidence = pitch_result
        if name != self.current_name:
            self.current_name = name
            self.smoothed_cents = cents
        else:
            self.smoothed_cents += (cents - self.smoothed_cents) * self.smoothing
        return name, self.smoothed_cents

    def draw(self, frame, pitch_result):
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        if self.update(pitch_result) is not None:
            self._last_heard = now
        showing = now - self._last_heard < HOLD_SECONDS
        self.visible += (FADE_IN if showing else -FADE_OUT) * dt
        self.visible = min(1.0, max(0.0, self.visible))
        if self.visible <= 0 or self.current_name is None:
            return
        self._render(frame, self.current_name, self.smoothed_cents, self.visible)

    def _render(self, frame, name, cents, visible):
        h, w = frame.shape[:2]
        color = cents_color(cents)
        in_tune = abs(cents) <= IN_TUNE_CENTS

        # Only the band of the frame the tuner (and its glow) covers
        half = int(LINE_HALF * w)
        pad = int(0.05 * h)
        x0, x1 = max(0, w // 2 - half - pad), min(w, w // 2 + half + pad)
        y1 = min(h, int(LINE_Y * h) + pad)
        roi = frame[0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        cx = w // 2 - x0
        lit = _Light((rh, rw))

        # The note
        text = _text_mask(name, max(10, int(NOTE_SIZE * h)))
        th, tw = text.shape
        tx, ty = int(cx - tw / 2), max(0, int(NOTE_Y * h - th / 2))
        note = np.zeros((rh, rw), np.float32)
        note[ty:ty + th, tx:tx + tw] = text[:rh - ty, :rw - tx]
        lit.add(note * 0.9, 0.013 * h, 1.2, core_white=0.6)
        # A soft shadow under it, so it reads on a bright picture
        shadow = 0.35 * np.minimum(1, _soften(note, 0.006 * h) * 2)

        # The line, with a tick at its centre, and the spot riding on it
        y = int(LINE_Y * h)
        line = np.zeros((rh, rw), np.float32)
        cv2.line(line, (cx - half, y), (cx + half, y), LINE_LIT if in_tune else LINE_DIM, 2, cv2.LINE_AA)
        tick = int(0.0075 * h)
        cv2.line(line, (cx, y - tick), (cx, y + tick), 0.5, 2, cv2.LINE_AA)
        lit.add(line, 0.0055 * h, 2.0, core_white=0.3)
        offset = max(-1.0, min(1.0, cents / RANGE_CENTS))
        x, spot_half = int(cx + offset * half), int(SPOT_HALF * h)
        spot = np.zeros((rh, rw), np.float32)
        cv2.line(spot, (x - spot_half, y), (x + spot_half, y), 1.0, 6, cv2.LINE_AA)
        lit.add(spot, 0.017 * h, 1.5, core_white=0.7)

        keep = cv2.merge([1 - visible * shadow] * 3)
        out = roi.astype(np.float32) * keep + visible * lit.bgr(color)
        roi[:] = np.clip(out, 0, 255).astype(np.uint8)
