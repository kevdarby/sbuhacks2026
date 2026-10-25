import cv2
import numpy as np


def glow(frame, draw_fn, ksize=31, intensity=1.0):
    """Draws onto a blank overlay via draw_fn(overlay), blurs it into a soft
    halo, and additively blends that halo behind the frame - the standard
    "bright core + blurred halo" trick for a neon look in plain OpenCV."""
    overlay = np.zeros_like(frame)
    draw_fn(overlay)
    blurred = cv2.GaussianBlur(overlay, (ksize, ksize), 0)
    if intensity != 1.0:
        blurred = (blurred.astype(np.float32) * intensity).astype(np.uint8)
    cv2.add(frame, blurred, dst=frame)
