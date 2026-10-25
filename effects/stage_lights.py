"""Coloured LED light spilling onto the picture from either side, as if
lights stood just out of frame: red from the right for the bass, blue from
the left for the drums. Each side glows faintly all the time and flares up
on every note its instrument plays. A third light, from the top, shows the
loop pedal's state (see effects/loop_border.py).

The light is applied as light, not painted on: the picture is multiplied
by it, so whatever's in frame picks up the colour in proportion to how
bright it already is (and dark areas stay dark), plus a little haze where
the light is strongest. The rest of the room is dimmed a touch, like house
lights turned down, so the LEDs dominate the way they would in person.

The band's threads only drop timestamped events into a BeatEvents queue;
the video loop releases each one when its note actually sounds. All times
are time.perf_counter(), the clock the drums and bass are scheduled on.
"""

import collections
import time

import cv2
import numpy as np

from drums.follower import DEFAULT_LEAD

BASS_COLOR = (40, 40, 255)    # BGR red
DRUM_COLOR = (255, 120, 30)   # BGR blue
# How lit each side is between notes, and how fast a flare dies away
IDLE = 0.2
DECAY = 0.22  # seconds for a flare to fall by ~2/3
# How far in the light reaches, as a fraction of the frame width: at idle
# and at a full flare (brighter light visibly carries further)
REACH_IDLE = 0.18
REACH_FULL = 0.38
# The room's own light, as a fraction of what the camera sees
AMBIENT = 0.8
# How strongly the light brightens what it falls on; how far it pulls what
# it falls on toward its own colour (so a white wall turns red, not pink);
# and the haze it adds
GAIN = 1.4
TINT = 0.7
HAZE = 0.25
# The light from the top (the loop pedal's): how far down it reaches
TOP_REACH = 0.4
# The light map is worked out at this fraction of the frame size - it's all
# smooth gradients - then scaled up
MAP_SCALE = 0.125


class BeatEvents:
    """Notes the band has played or scheduled, as (sound time, kind,
    strength). Appended from the drum and bass threads, drained from the
    video loop."""

    def __init__(self):
        self._queue = collections.deque()
        self._pending = []

    def add(self, sound_time, kind, strength=1.0):
        self._queue.append((sound_time, kind, strength))  # deque.append is thread-safe

    def due(self, now):
        """Events whose note has sounded by `now`, oldest first. Stale ones
        (the video stalled) are dropped rather than flaring all at once."""
        while self._queue:
            self._pending.append(self._queue.popleft())
        ready = [e for e in self._pending if e[0] <= now]
        self._pending = [e for e in self._pending if e[0] > now]
        return [e for e in sorted(ready) if now - e[0] < 0.25]


class TappedRelay:
    """Stands in front of a Relay (or SoftwareRelay) and reports each hit, at
    the time it will sound, to a BeatEvents - so live drums and loop replay
    both flash the lights without either knowing about the video."""

    def __init__(self, relay, events, lead=DEFAULT_LEAD):
        self._relay = relay
        self._events = events
        self._lead = lead
        self.port = relay.port

    def hit(self, pulse):
        self._events.add(time.perf_counter() + self._lead, "drums", max(0.5, min(1.0, pulse / 0.08)))
        self._relay.hit(pulse)

    def close(self):
        self._relay.close()


class StageLights:
    """For main.py: call draw() once a frame."""

    def __init__(self):
        self.events = BeatEvents()
        self.level = {"bass": IDLE, "drums": IDLE}
        self.top = None  # (colour, level) of the light from the top, or None
        # Off: no side lights (and their notes are ignored); the top light
        # still shows
        self.sides = True

    def on_bass_note(self, start, velocity):
        self.events.add(start, "bass", max(0.4, min(1.0, velocity)))

    def _light(self, side, level, w, h):
        """One side's light on the small map: falling off from its edge,
        reaching further the brighter it is, and a little stronger toward
        the middle of the frame's height, where the lights would point."""
        x = np.linspace(0, 1, w, dtype=np.float32)
        if side == "bass":
            x = 1 - x
        reach = REACH_IDLE + (REACH_FULL - REACH_IDLE) * (level - IDLE) / (1 - IDLE)
        across = np.exp(-(x / reach) ** 1.5)
        y = np.linspace(-1, 1, h, dtype=np.float32)
        down = 0.7 + 0.3 * np.exp(-(y / 0.8) ** 2)
        return level * down[:, None] * across[None, :]

    def _top_light(self, level, w, h):
        """Light from above: falling off down the frame, strongest across
        the middle."""
        y = np.linspace(0, 1, h, dtype=np.float32)
        down = np.exp(-(y / TOP_REACH) ** 1.5)
        x = np.linspace(-1, 1, w, dtype=np.float32)
        across = 0.5 + 0.5 * np.exp(-(x / 0.7) ** 2)
        return level * down[:, None] * across[None, :]

    def draw(self, frame, dt):
        fall = np.exp(-dt / DECAY)
        for kind in self.level:
            self.level[kind] = IDLE + (self.level[kind] - IDLE) * fall
        for _, kind, strength in self.events.due(time.perf_counter()):
            self.level[kind] = max(self.level[kind], strength)
        if not self.sides:
            self.level = {kind: IDLE for kind in self.level}

        h, w = frame.shape[:2]
        sw, sh = max(2, int(w * MAP_SCALE)), max(2, int(h * MAP_SCALE))
        # What each pixel's colour channels get multiplied by, and the haze
        factor = np.full((sh, sw, 3), AMBIENT, np.float32)
        haze = np.zeros((sh, sw, 3), np.float32)
        lights = [(self._light(kind, self.level[kind], sw, sh), color)
                  for kind, color in (("bass", BASS_COLOR), ("drums", DRUM_COLOR))
                  if self.sides]
        if self.top is not None:
            lights.append((self._top_light(self.top[1], sw, sh), self.top[0]))
        if not lights:
            return  # the plain picture, with the room's light left alone too
        for light, color in lights:
            m = np.minimum(1, light)[..., None]
            c = np.asarray(color, np.float32) / 255
            # Pull the channels the light lacks down, push the ones it has up
            factor *= 1 - m * TINT * (1 - c)
            factor += m * GAIN * c
            haze += m * HAZE * 255 * c

        # factor runs up to ~2.5; stored as factor/4 in a byte, so the
        # multiply's 4/255 scale gets back picture x factor
        factor = cv2.resize(np.clip(factor * 255 / 4, 0, 255).astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_LINEAR)
        haze = cv2.resize(np.clip(haze, 0, 255).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_LINEAR)
        cv2.multiply(frame, factor, dst=frame, scale=4 / 255)
        cv2.add(frame, haze, dst=frame)
