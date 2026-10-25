"""Hand-based volume gesture: openness ratio + palm-orientation freeze."""

# Landmarks: 0 = wrist, these = the five fingertips (mediapipe Hands topology).
FINGERTIPS = [4, 8, 12, 16, 20]
# Wrist-to-middle-MCP distance is roughly constant regardless of how open the
# hand is, so it's used to normalize fingertip spread against hand size/
# distance from the camera.
PALM_REF = 9
# Index-MCP and pinky-MCP, used with the wrist to detect palm orientation.
INDEX_MCP = 5
PINKY_MCP = 17


def hand_openness_ratio(landmarks):
    wrist = landmarks[0]
    palm_ref = landmarks[PALM_REF]
    palm_size = ((wrist.x - palm_ref.x) ** 2 + (wrist.y - palm_ref.y) ** 2) ** 0.5
    if palm_size < 1e-6:
        return None

    spread = 0.0
    for idx in FINGERTIPS:
        tip = landmarks[idx]
        spread += ((wrist.x - tip.x) ** 2 + (wrist.y - tip.y) ** 2) ** 0.5

    return spread / (palm_size * len(FINGERTIPS))


def palm_facing_signal(landmarks, handedness="Right"):
    """Normalized (sin-of-angle) 2D cross product of (wrist->index_mcp) x
    (wrist->pinky_mcp), in [-1, 1], positive when the palm faces the camera.
    Its sign flips when the hand rotates ~180 degrees around the vertical
    axis (palm toward vs. away from camera), and normalizing by vector
    lengths keeps it independent of hand distance/size - used to freeze
    volume updates when you turn your hand away instead of tracking an
    unreliable edge-on hand shape.

    A left hand is a mirror image of a right one, so the same palm-forward
    pose winds the other way: without `handedness` (MediaPipe's "Left" /
    "Right" label) a left palm facing the camera read as turned away."""
    wrist = landmarks[0]
    index_mcp = landmarks[INDEX_MCP]
    pinky_mcp = landmarks[PINKY_MCP]
    v1x, v1y = index_mcp.x - wrist.x, index_mcp.y - wrist.y
    v2x, v2y = pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y
    n1 = (v1x ** 2 + v1y ** 2) ** 0.5
    n2 = (v2x ** 2 + v2y ** 2) ** 0.5
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    signal = (v1x * v2y - v1y * v2x) / (n1 * n2)
    return -signal if handedness == "Left" else signal


class PalmFacingGate:
    """Tracks facing/away state with hysteresis around 0 so noise near the
    edge-on transition doesn't flicker the state every frame."""

    def __init__(self, threshold=0.15):
        self.threshold = threshold
        self.facing = True

    def update(self, signal):
        if signal > self.threshold:
            self.facing = True
        elif signal < -self.threshold:
            self.facing = False
        return self.facing
