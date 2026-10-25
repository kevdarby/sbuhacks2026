import cv2

from effects.glow import glow


def draw_trigger_burst(frame, center, age, color=(255, 255, 255),
                        duration=0.35, max_radius=90):
    """Expanding, fading ring - the 'click' feedback for the instant the
    mouth trigger fires. age = seconds since it fired; call every frame
    while age < duration."""
    if age < 0 or age > duration:
        return
    t = age / duration
    radius = int(10 + max_radius * t)

    def draw_ring(canvas):
        cv2.circle(canvas, center, radius, color, max(2, int(6 * (1 - t))),
                   cv2.LINE_AA)

    glow(frame, draw_ring, ksize=21, intensity=1.0 - t)
    cv2.circle(frame, center, radius, color, max(1, int(3 * (1 - t))),
               cv2.LINE_AA)
