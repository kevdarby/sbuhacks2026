"""The camera overlay, styled after the website branch's design (see
effects/comic.py): brand and status sticker, a "playing strip" with tempo,
beat and chord, paper cards for the drums and bass, a magenta loop panel,
a tuner sticker, and gesture rings - everything main.py shows, in one
place. main.py fills in a HudState each frame; Hud.draw paints it.

Laid out on a 1920x1080 grid and scaled to the actual frame width.
"""

import math
from dataclasses import dataclass, field

import cv2

from amp.audio import IDLE, PLAYING, RECORDING
from effects.comic import (ACCENT, CYAN, INK, LINE, MUTED, PAGE, PAPER, PAPER_TEXT, PINK,
                           SUB, Shade, bar, blit, diamond, panel, starburst, tag, text,
                           text_sprite)

IN_TUNE_CENTS = 5
CLOSE_CENTS = 20


@dataclass
class Ring:
    """A hand's gesture ring."""
    kind: str            # "bass" or "drums"
    center: tuple
    radius: int
    percent: float
    pop: float = 1.0


@dataclass
class HudState:
    now: float = 0.0
    status: str = "listening"     # listening, locking, live, loop
    bpm: float = None
    beat: int = None              # 0-3 within the bar, while the band plays
    step: int = None              # 0-15 within the bar
    chord: str = None
    drums_on: bool = True
    drum_pattern: str = None      # 16 steps, drums/patterns.py notation
    drum_feel: str = None
    drum_percent: float = None    # None = automatic
    bass_on: bool = True
    bass_note: str = None
    bass_style: str = None
    bass_percent: float = 50.0
    bass_learned: bool = False
    loop_state: str = IDLE
    loop_bars: int = None
    loop_seconds: float = 0.0
    loop_gain_db: float = 0.0
    loop_output: str = "amp"
    tuner: tuple = None           # (note name, cents) or None
    rings: list = field(default_factory=list)
    burst: tuple = None           # (center, age seconds, word)
    messages: list = field(default_factory=list)   # (text, BGR color)
    diagnostics: str = None


def cents_color(cents):
    if abs(cents) <= IN_TUNE_CENTS:
        return CYAN
    if abs(cents) <= CLOSE_CENTS:
        return ACCENT
    return PINK


class Hud:
    def __init__(self):
        self.shade = Shade()

    def draw(self, frame, s):
        h, w = frame.shape[:2]
        self.k = w / 1920  # layout scale
        self.shade.apply(frame)
        self._loop_frame(frame, s)
        self._masthead(frame, s)
        self._playing_strip(frame, s)
        self._tuner(frame, s)
        for ring in s.rings:
            self._ring(frame, ring)
        cards_top = self._band_cards(frame, s)
        self._loop_panel(frame, s)
        self._messages(frame, s, cards_top)
        if s.burst is not None:
            self._burst(frame, *s.burst)
        if s.diagnostics:
            text(frame, w // 2, int(14 * self.k), s.diagnostics, "mono_light", self.px(15), SUB,
                 shadows=[(1, 1, PAGE)], anchor="c")

    def px(self, v):
        return max(1, int(round(v * self.k)))

    # --- top left ---------------------------------------------------------

    def _masthead(self, frame, s):
        k = self.px
        x, y = k(48), k(30)
        _, _, x1, _ = text(frame, x, y, "BONGOS", "display", k(70), INK,
                           shadows=[(k(3), k(3), PINK)], angle=3, tracking=-1, tail=(".", ACCENT))
        text(frame, x1 + k(36), y + k(24), "VOL. 01 / LIVE", "mono", k(17), CYAN, tracking=2)

        label = {"listening": "PLUG IN. TUNE OUT.", "locking": "LOCKING ON",
                 "live": "LIVE", "loop": "ON REPEAT"}[s.status]
        fill = {"listening": CYAN, "locking": ACCENT, "live": CYAN, "loop": ACCENT}[s.status]
        tag(frame, x, k(126), label, fill, size=k(19), shadow=PINK, angle=2,
            pad=(k(14), k(8)), dot=True)

    def _playing_strip(self, frame, s):
        k = self.px
        x0, y0, x1, y1 = k(48), k(196), k(628), k(318)
        panel(frame, x0, y0, x1, y1, PAGE, border=PAPER, border_px=k(2), shadow=(8, 8, 8),
              shadow_px=k(6))
        bpm = "—" if s.bpm is None else f"{s.bpm:.0f}"
        _, _, bx1, _ = text(frame, x0 + k(24), y0 + k(14), bpm, "display", k(64),
                            MUTED if s.bpm is None else INK)
        text(frame, bx1 + k(8), y0 + k(56), "BPM", "mono", k(15), CYAN)
        # Beat diamonds, the current beat lit
        for i in range(4):
            lit = s.beat == i
            diamond(frame, x0 + k(230) + i * k(34), y0 + k(54), k(11),
                    ACCENT if lit else MUTED, shadow=PINK if lit else None)
        _, _, cx1, _ = text(frame, x0 + k(390), y0 + k(14), s.chord or "—", "display", k(64),
                            INK if s.chord else MUTED)
        text(frame, cx1 + k(8), y0 + k(56), "CHORD", "mono", k(15), CYAN)
        line = {"listening": "Start playing - the band follows your lead.",
                "locking": "Hearing your tempo...",
                "live": "The band is following you.",
                "loop": "Replaying the band's take with your loop."}[s.status]
        text(frame, x0 + k(24), y1 - k(36), line, "mono_light", k(15), SUB)

    # --- top right --------------------------------------------------------

    def _tuner(self, frame, s):
        k = self.px
        w = frame.shape[1]
        x1, y0 = w - k(48), k(36)
        x0, y1 = x1 - k(250), y0 + k(150)
        panel(frame, x0, y0, x1, y1, PAPER, shadow=CYAN, shadow_px=k(7), border_px=k(3))
        tag(frame, x0 + k(16), y0 + k(14), "TUNER", CYAN, size=k(15), pad=(k(8), k(4)))
        cx, ny = (x0 + x1) // 2, y0 + k(105)
        if s.tuner is None:
            text(frame, cx, y0 + k(62), "—", "display", k(52), MUTED, anchor="c")
        else:
            name, cents = s.tuner
            color = cents_color(cents)
            text(frame, cx, y0 + k(64), name, "display", k(54), PAGE,
                 shadows=[(k(3), k(3), color)], anchor="c")
        cv2.line(frame, (x0 + k(24), ny + k(14)), (x1 - k(24), ny + k(14)), PAGE, k(3))
        cv2.line(frame, (cx, ny + k(4)), (cx, ny + k(24)), PAGE, k(3))
        if s.tuner is not None:
            span = (x1 - x0 - k(48)) / 2
            offset = max(-50, min(50, s.tuner[1])) / 50
            diamond(frame, int(cx + offset * span), ny + k(14), k(11), cents_color(s.tuner[1]),
                    shadow=PAGE)
        if s.status in ("live", "loop"):
            # The web's gig sticker
            sx, sy = x1 - k(226), y1 + k(34)
            panel(frame, sx, sy, sx + k(214), sy + k(92), PINK, shadow=CYAN, shadow_px=k(6),
                  angle=-6)
            text(frame, sx + k(18), sy + k(12), "LIVE FROM", "mono", k(13), PAGE, angle=-6)
            text(frame, sx + k(14), sy + k(32), "YOUR BEDROOM.", "display", k(30), PAGE, angle=-6)

    # --- bottom -----------------------------------------------------------

    def _band_cards(self, frame, s):
        k = self.px
        h = frame.shape[0]
        y1 = h - k(42)
        y0 = y1 - k(262)
        self._drum_card(frame, s, k(48), y0, k(468), y1)
        self._bass_card(frame, s, k(504), y0, k(924), y1)
        return y0

    def _card_header(self, frame, x0, y0, x1, label, title, accent, on, status):
        k = self.px
        tag(frame, x0 + k(22), y0 + k(18), label, accent, size=k(14), pad=(k(8), k(4)))
        text(frame, x0 + k(20), y0 + k(40), title, "display", k(62), PAGE)
        text(frame, x0 + k(24), y0 + k(122), status, "mono_light", k(15), PAPER_TEXT)
        # The web's toggle switch: on = filled with the part's color
        tx1, ty0 = x1 - k(26), y0 + k(62)
        tx0, ty1 = tx1 - k(50), ty0 + k(28)
        panel(frame, tx0, ty0, tx1, ty1, accent if on else (185, 198, 203), border_px=k(2),
              angle=-4)
        knob = tx1 - k(24) if on else tx0 + k(4)
        cv2.rectangle(frame, (knob, ty0 + k(4)), (knob + k(20), ty1 - k(4)),
                      PAGE if on else PAPER, -1)

    def _drum_card(self, frame, s, x0, y0, x1, y1):
        k = self.px
        panel(frame, x0, y0, x1, y1, PAPER, shadow=CYAN, shadow_px=k(8), border_px=k(3),
              angle=0.6)
        if not s.drums_on:
            status = "Silent"
        elif s.drum_feel:
            status = s.drum_feel
        else:
            status = "Waiting for you"
        self._card_header(frame, x0, y0, x1, "01 / THE PULSE", "DRUMS", CYAN, s.drums_on,
                          status)
        # The groove, step by step, with a playhead
        gx0, gy, cell = x0 + k(24), y0 + k(156), (x1 - x0 - k(48)) / 16
        pattern = s.drum_pattern or "." * 16
        for i, symbol in enumerate(pattern[:16]):
            cx0 = int(gx0 + i * cell)
            cx1 = int(gx0 + (i + 1) * cell) - k(3)
            here = s.step == i
            if here:
                cv2.rectangle(frame, (cx0 - k(2), gy - k(4)), (cx1 + k(2), gy + k(34)), ACCENT, -1)
            if symbol != ".":
                top = gy + {"X": 0, "x": k(8), "g": k(16)}.get(symbol, k(8))
                cv2.rectangle(frame, (cx0, top), (cx1, gy + k(30)), PINK if here else PAGE, -1)
            else:
                cv2.rectangle(frame, (cx0, gy + k(26)), (cx1, gy + k(30)), (190, 190, 190), -1)
            if i % 4 == 0:
                cv2.line(frame, (cx0, gy + k(36)), (cx0, gy + k(42)), PAGE, k(2))
        value = "AUTO" if s.drum_percent is None else f"{s.drum_percent:.0f}%"
        text(frame, x0 + k(24), y1 - k(58), "INTENSITY", "mono", k(14), PAGE)
        text(frame, x1 - k(24), y1 - k(58), value, "mono_light", k(14), PAPER_TEXT, anchor="tr")
        bar(frame, x0 + k(24), y1 - k(32), x1 - k(24), y1 - k(20),
            0.5 if s.drum_percent is None else s.drum_percent / 100, PAGE)

    def _bass_card(self, frame, s, x0, y0, x1, y1):
        k = self.px
        panel(frame, x0, y0, x1, y1, PAPER, shadow=PINK, shadow_px=k(8), border_px=k(3),
              angle=-0.6)
        if not s.bass_on:
            status = "Off"
        elif s.loop_state == PLAYING and s.status == "loop":
            status = "In the loop recording"
        elif s.bass_note is None:
            status = "Waiting for a chord"
        else:
            status = f"{s.bass_style or ''} · {'knows the progression' if s.bass_learned else 'following'}"
        self._card_header(frame, x0, y0, x1, "02 / THE LOW END", "BASS", PINK, s.bass_on, status)
        if s.bass_on and s.bass_note:
            text(frame, x1 - k(30), y0 + k(112), s.bass_note, "display", k(64), PAGE,
                 shadows=[(k(3), k(3), PINK)], anchor="tr")
        text(frame, x0 + k(24), y1 - k(58), "LEVEL", "mono", k(14), PAGE)
        text(frame, x1 - k(24), y1 - k(58), f"{s.bass_percent:.0f}%", "mono_light", k(14),
             PAPER_TEXT, anchor="tr")
        bar(frame, x0 + k(24), y1 - k(32), x1 - k(24), y1 - k(20), s.bass_percent / 100, PAGE)

    def _loop_panel(self, frame, s):
        k = self.px
        h, w = frame.shape[:2]
        x1, y1 = w - k(56), h - k(50)
        x0, y0 = x1 - k(470), y1 - k(184)
        fill = PINK if s.loop_state != PLAYING else ACCENT
        panel(frame, x0, y0, x1, y1, fill, shadow=ACCENT if s.loop_state != PLAYING else PINK,
              shadow_px=k(8), border_px=k(3), angle=-0.7)
        tag(frame, x0 + k(24), y0 + k(20), "YOUR LOOP", PAPER, size=k(14), pad=(k(8), k(4)),
            angle=2)
        rec_dot = False
        if s.loop_state == RECORDING:
            rec_dot = math.sin(s.now * 7) > 0
            headline = "RECORDING"
            sub = f"{s.loop_seconds:4.1f}s · mouth / space to stop"
        elif s.loop_state == PLAYING:
            bars = f" · {s.loop_bars} BAR{'S' * (s.loop_bars != 1)}" if s.loop_bars else ""
            headline = "LOOPING" + bars
            sub = f"{s.loop_gain_db:+.0f}dB (+/-) · out: {s.loop_output} (t) · mouth to clear"
        else:
            headline = "CATCH THE RIFF."
            sub = "Open your mouth or press space to record."
        hx = x0 + k(22)
        if s.loop_state == RECORDING:
            if rec_dot:
                cv2.circle(frame, (hx + k(16), y0 + k(98)), k(15), PAGE, -1, cv2.LINE_AA)
            hx += k(44)
        text(frame, hx, y0 + k(56), headline, "display", k(58), PAGE)
        text(frame, x0 + k(26), y1 - k(42), sub, "mono_light", k(14), (45, 18, 33))

    def _loop_frame(self, frame, s):
        """A thick poster border around the whole frame while looping."""
        if s.loop_state == IDLE:
            return
        k = self.px
        h, w = frame.shape[:2]
        if s.loop_state == RECORDING:
            pulse = (math.sin(s.now * 7) + 1) / 2
            outer, inner, t = PINK, CYAN, k(12 + 8 * pulse)
        else:
            outer, inner, t = ACCENT, PINK, k(12)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), outer, t * 2)
        cv2.rectangle(frame, (t + k(4), t + k(4)), (w - t - k(5), h - t - k(5)), inner, k(3))

    def _messages(self, frame, s, cards_top):
        k = self.px
        y = cards_top - k(26)
        for words, color in reversed(s.messages):
            # On a dark label so they read over a bright room too
            sprite = text_sprite(words, "mono", k(16), color)
            sh, sw = sprite.shape[:2]
            panel(frame, k(48), y - sh - k(10), k(48) + sw + k(20), y, PAGE, border=LINE,
                  border_px=k(1))
            blit(frame, sprite, k(58), y - k(5), anchor="bl")
            y -= sh + k(18)

    # --- floating -------------------------------------------------------

    def _ring(self, frame, ring):
        k = self.px
        cx, cy = ring.center
        r = ring.radius
        color, other = (PINK, CYAN) if ring.kind == "bass" else (CYAN, PINK)
        thick = k(14)
        cv2.circle(frame, (cx, cy), r, PAGE, thick + k(6), cv2.LINE_AA)
        end = -90 + 360 * min(max(ring.percent, 0), 100) / 100
        if ring.percent > 0:
            cv2.ellipse(frame, (cx + k(6), cy + k(6)), (r, r), 0, -90, end, other, thick,
                        cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, end, color, thick, cv2.LINE_AA)
        size = int(k(54) * ring.pop)
        text(frame, cx, cy, f"{ring.percent:.0f}", "display", size, INK,
             shadows=[(k(3), k(3), color), (k(5), k(5), PAGE)], anchor="c")
        label = "BASS" if ring.kind == "bass" else "DRUMS"
        sprite = text_sprite(label, "mono", k(16), PAGE, tracking=1)
        tw = sprite.shape[1] + k(16)
        tag(frame, cx - tw // 2, cy + r + k(14), label, color, size=k(16), shadow=PAGE,
            angle=-3 if ring.kind == "bass" else 3, pad=(k(8), k(4)))

    def _burst(self, frame, center, age, word):
        if age > 0.6:
            return
        k = self.px
        grow = 0.6 + 0.4 * min(age / 0.12, 1)
        r = int(k(86) * grow)
        cx, cy = center
        starburst(frame, cx, cy, r, ACCENT, shadow=PINK, seed=len(word))
        text(frame, cx, cy, word, "display", int(k(44) * grow), PAGE, anchor="c", angle=4)
