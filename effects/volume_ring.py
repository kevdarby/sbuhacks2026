import cv2
import numpy as np

from effects.glow import glow


def volume_color(percent):
    """Green at 0, through yellow, to red at 100 - an intuitive "how loud"
    color ramp, done via HSV hue interpolation."""
    hue = int(max(0, 75 - 0.75 * percent))  # 75 (green) at 0% -> 0 (red) at 100%
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class PulseValue:
    """Smoothly animates toward a target value instead of snapping, so
    on-screen numbers/sizes glide rather than jump - and exposes how
    recently the target changed, for a "pop" bump on change."""

    def __init__(self, initial=0.0, speed=12.0):
        self.value = initial
        self.speed = speed
        self._last_target = initial
        self.changed_at = -999.0

    def update(self, target, now, dt):
        if target is not None and abs(target - self._last_target) > 0.5:
            self.changed_at = now
            self._last_target = target
        if target is not None:
            self.value += (target - self.value) * min(1.0, self.speed * dt)
        return self.value

    def pop_scale(self, now, duration=0.15, magnitude=0.35):
        age = now - self.changed_at
        if age < 0 or age > duration:
            return 1.0
        return 1.0 + magnitude * (1 - age / duration)


def draw_volume_ring(frame, center, radius, percent, pop_scale=1.0, thickness=10, caption=None):
    color = volume_color(percent)
    cx, cy = center

    # dim background track
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 360,
                (55, 55, 55), thickness // 2, cv2.LINE_AA)

    start_angle = -90
    end_angle = start_angle + 360 * (percent / 100)

    def draw_arc(canvas):
        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, start_angle,
                    end_angle, color, thickness + 6, cv2.LINE_AA)

    glow(frame, draw_arc, ksize=25, intensity=0.9)
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle,
                color, thickness, cv2.LINE_AA)

    label = f"{round(percent)}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.3 * pop_scale
    (tw, th), _ = cv2.getTextSize(label, font, scale, 3)
    org = (cx - tw // 2, cy + th // 2)
    cv2.putText(frame, label, org, font, scale, (10, 10, 10), 5, cv2.LINE_AA)
    cv2.putText(frame, label, org, font, scale, (255, 255, 255), 2, cv2.LINE_AA)

    if caption:
        # What the ring controls, small, under the number
        cap_scale = 0.55
        (cw, ch), _ = cv2.getTextSize(caption, font, cap_scale, 1)
        cap_org = (cx - cw // 2, cy + th // 2 + ch + 12)
        cv2.putText(frame, caption, cap_org, font, cap_scale, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(frame, caption, cap_org, font, cap_scale, color, 1, cv2.LINE_AA)
