"""Comic gig-poster drawing kit for the HUD, after the web design on the
website branch (web/styles.css): midnight indigo, hot magenta, electric
cyan, acid yellow and cream "paper", hard offset shadows instead of glows,
tilted stickers, Barlow Condensed Black Italic headlines over IBM Plex
Mono labels (assets/fonts, OFL - Impact/Menlo stand in if missing).

OpenCV's own fonts can't do any of that, so text is rendered with Pillow
into small RGBA sprites, cached by content - a frame only re-renders text
whose words changed - and alpha-blended onto the BGR camera frame.
"""

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Palette from web/styles.css, as BGR
PAGE = (38, 17, 17)        # #111126 midnight indigo
SURFACE = (53, 26, 27)     # #1b1a35
PAPER = (223, 246, 255)    # #fff6df cream
INK = (232, 248, 255)      # #fff8e8
SUB = (223, 203, 208)      # #d0cbdf
LINE = (122, 97, 103)      # #67617a
ACCENT = (69, 255, 239)    # #efff45 acid yellow
PINK = (152, 50, 255)      # #ff3298 hot magenta
CYAN = (255, 228, 101)     # #65e4ff electric cyan
MUTED = (117, 96, 100)     # #646075 unlit beat dot
PAPER_TEXT = (89, 70, 81)  # #51465d secondary text on paper

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
DISPLAY_FONTS = [FONT_DIR / "BarlowCondensed-BlackItalic.ttf",
                 Path("/System/Library/Fonts/Supplemental/Impact.ttf")]
MONO_FONTS = [FONT_DIR / "IBMPlexMono-Bold.ttf", Path("/System/Library/Fonts/Menlo.ttc")]
MONO_LIGHT_FONTS = [FONT_DIR / "IBMPlexMono-Medium.ttf", Path("/System/Library/Fonts/Menlo.ttc")]
FONTS = {"display": DISPLAY_FONTS, "mono": MONO_FONTS, "mono_light": MONO_LIGHT_FONTS}


@lru_cache(maxsize=64)
def _font(kind, size):
    for path in FONTS[kind]:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def _rgba(bgr):
    return (bgr[2], bgr[1], bgr[0], 255)


_sprites = OrderedDict()
SPRITE_CACHE_SIZE = 400


def text_sprite(text, kind, size, color, shadows=(), angle=0.0, tracking=0, tail=None):
    """RGBA (as BGRA numpy) image of `text`, with hard offset shadows
    [(dx, dy, color), ...] drawn behind it like the web headlines, rotated
    by `angle` degrees (counter-clockwise). `tail` = (text, color) is
    appended in another color, like the brand's yellow period. Cached.

    Unrotated text keeps the font's full line height, so text drawn at the
    same y shares a baseline - cropping to the ink made a lone dash sit at
    the top of the line."""
    key = (text, kind, size, color, tuple(shadows), round(angle, 1), tracking, tail)
    sprite = _sprites.get(key)
    if sprite is not None:
        _sprites.move_to_end(key)
        return sprite
    font = _font(kind, size)
    pad = size // 2 + max([max(abs(dx), abs(dy)) for dx, dy, _ in shadows] + [0])
    if tracking:
        widths = [font.getlength(ch) + tracking for ch in text]
        width = int(sum(widths)) + 2 * pad
    else:
        width = int(font.getlength(text)) + 2 * pad
    if tail:
        width += int(font.getlength(tail[0]))
    ascent, descent = font.getmetrics()
    image = Image.new("RGBA", (max(width, 1), ascent + descent + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def put(x, y, fill):
        if tracking:
            for ch, w in zip(text, widths):
                draw.text((x, y), ch, font=font, fill=fill)
                x += w
        else:
            draw.text((x, y), text, font=font, fill=fill)

    tail_x = pad + (sum(widths) if tracking else font.getlength(text))
    for dx, dy, shadow in shadows:
        put(pad + dx, pad + dy, _rgba(shadow))
        if tail:
            draw.text((tail_x + dx, pad + dy), tail[0], font=font, fill=_rgba(shadow))
    put(pad, pad, _rgba(color))
    if tail:
        draw.text((tail_x, pad), tail[0], font=font, fill=_rgba(tail[1]))
    if angle:
        image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
    else:
        bbox = image.getbbox()
        if bbox:
            # Trim sideways to the ink, but keep the line's full height
            image = image.crop((bbox[0], pad, bbox[2], max(bbox[3], pad + ascent + descent)))
    sprite = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
    _sprites[key] = sprite
    if len(_sprites) > SPRITE_CACHE_SIZE:
        _sprites.popitem(last=False)
    return sprite


def blit(frame, sprite, x, y, anchor="tl"):
    """Alpha-blend a BGRA sprite onto the frame. anchor: tl, tr, bl, br, c,
    or ml (middle-left). Returns the drawn (x0, y0, x1, y1)."""
    sh, sw = sprite.shape[:2]
    if anchor[0] == "b":
        y -= sh
    elif anchor[0] in "cm":
        y -= sh // 2
    if anchor[-1] == "r":
        x -= sw
    elif anchor == "c":
        x -= sw // 2
    fh, fw = frame.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + sw, fw), min(int(y) + sh, fh)
    if x1 <= x0 or y1 <= y0:
        return x0, y0, x0, y0
    part = sprite[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)]
    alpha = part[:, :, 3:4].astype(np.uint16)
    roi = frame[y0:y1, x0:x1]
    roi[:] = ((part[:, :, :3] * alpha + roi * (255 - alpha)) // 255).astype(np.uint8)
    return x0, y0, x1, y1


def text(frame, x, y, words, kind="mono", size=20, color=INK, shadows=(), angle=0.0,
         anchor="tl", tracking=0, tail=None):
    sprite = text_sprite(words, kind, size, color, shadows, angle, tracking, tail)
    return blit(frame, sprite, x, y, anchor)


def _rotated(points, angle, center):
    if not angle:
        return np.array(points, dtype=np.int32)
    c, s = np.cos(np.radians(-angle)), np.sin(np.radians(-angle))
    cx, cy = center
    return np.array([(cx + (px - cx) * c - (py - cy) * s, cy + (px - cx) * s + (py - cy) * c)
                     for px, py in points], dtype=np.int32)


def panel(frame, x0, y0, x1, y1, fill, border=PAGE, border_px=3, shadow=None, shadow_px=7,
          angle=0.0):
    """A flat poster panel with a thick border and a hard offset shadow
    (bottom-right), optionally tilted a few degrees."""
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    if shadow is not None:
        cv2.fillPoly(frame, [_rotated([(px + shadow_px, py + shadow_px) for px, py in corners],
                                      angle, center)], shadow, cv2.LINE_AA)
    poly = _rotated(corners, angle, center)
    cv2.fillPoly(frame, [poly], fill, cv2.LINE_AA)
    if border_px:
        cv2.polylines(frame, [poly], True, border, border_px, cv2.LINE_AA)


def tag(frame, x, y, words, fill, color=PAGE, size=18, shadow=None, angle=0.0, pad=(10, 6),
        anchor="tl", dot=False):
    """A small mono label on a solid colored block - the web's eyebrows and
    '01 / THE PULSE' tags. dot: a status dot before the words (drawn - the
    fonts have no bullet glyph)."""
    sprite = text_sprite(words, "mono", size, color, tracking=1)
    sh, sw = sprite.shape[:2]
    dot_w = int(size * 1.1) if dot else 0
    w, h = sw + 2 * pad[0] + dot_w, sh + 2 * pad[1]
    if anchor[-1] == "r":
        x -= w
    if anchor[0] == "b":
        y -= h
    panel(frame, x, y, x + w, y + h, fill, border=PAGE, border_px=2, shadow=shadow,
          shadow_px=4, angle=angle)
    if dot:
        cv2.circle(frame, (x + pad[0] + size // 3, y + h // 2), max(2, size // 4), color, -1,
                   cv2.LINE_AA)
    blit(frame, sprite, x + pad[0] + dot_w, y + pad[1])
    return x, y, x + w, y + h


def diamond(frame, cx, cy, r, fill, shadow=None):
    pts = np.array([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], dtype=np.int32)
    if shadow is not None:
        cv2.fillPoly(frame, [pts + 3], shadow, cv2.LINE_AA)
    cv2.fillPoly(frame, [pts], fill, cv2.LINE_AA)


def bar(frame, x0, y0, x1, y1, fraction, fill, track=(203, 198, 185), border=PAGE):
    """A flat horizontal meter, like the web's range inputs."""
    cv2.rectangle(frame, (x0, y0), (x1, y1), track, -1)
    fraction = min(max(fraction, 0.0), 1.0)
    if fraction > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + int((x1 - x0) * fraction), y1), fill, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)


def starburst(frame, cx, cy, r, fill, shadow=None, points=12, seed=0):
    """A jagged comic 'POW' burst."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(points * 2):
        ang = np.pi * i / points
        rad = r if i % 2 == 0 else r * rng.uniform(0.5, 0.65)
        pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
    pts = np.array(pts, dtype=np.int32)
    if shadow is not None:
        cv2.fillPoly(frame, [pts + 6], shadow, cv2.LINE_AA)
    cv2.fillPoly(frame, [pts], fill, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, PAGE, 3, cv2.LINE_AA)


class Shade:
    """The web hero's indigo fade over the art - here over the camera image,
    from the left edge and the bottom, so the HUD reads over anything.
    Built once per frame size as integer multiply/add images, so applying
    it is two fast OpenCV ops rather than float blending a 1080p frame."""

    def __init__(self, left=0.40, left_strength=0.62, bottom=0.36, bottom_strength=0.78):
        self.params = (left, left_strength, bottom, bottom_strength)
        self._size = None

    def _build(self, h, w):
        left, ls, bottom, bs = self.params
        xs = np.clip(1 - np.arange(w) / (left * w), 0, 1) ** 1.6 * ls
        ys = np.clip(1 - (h - np.arange(h)) / (bottom * h), 0, 1) ** 1.4 * bs
        alpha = 1 - (1 - xs[None, :]) * (1 - ys[:, None])
        self._keep = np.repeat((255 * (1 - alpha))[:, :, None], 3, axis=2).astype(np.uint8)
        self._add = (alpha[:, :, None] * np.array(PAGE)[None, None, :]).astype(np.uint8)
        self._size = (h, w)

    def apply(self, frame):
        h, w = frame.shape[:2]
        if self._size != (h, w):
            self._build(h, w)
        cv2.multiply(frame, self._keep, dst=frame, scale=1 / 255)
        cv2.add(frame, self._add, dst=frame)
