"""Mouth-open gesture for fast, pedal-style loop triggering. Unlike a
static hand pose, a face gesture is a deliberate binary event and doesn't
need a hold-time to distinguish it from a hand passing through a shape
mid-motion - so this fires almost instantly (a couple of frames' debounce).
"""

# mediapipe Face Mesh landmark indices: inner lip top/bottom (mouth-open
# distance) and outer eye corners (stable face-scale reference, unaffected
# by mouth movement - same normalization trick as the hand's palm_ref).
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
FACE_SCALE_LEFT = 33
FACE_SCALE_RIGHT = 263


def mouth_openness_ratio(face_landmarks):
    top = face_landmarks[MOUTH_TOP]
    bottom = face_landmarks[MOUTH_BOTTOM]
    left = face_landmarks[FACE_SCALE_LEFT]
    right = face_landmarks[FACE_SCALE_RIGHT]
    scale = ((left.x - right.x) ** 2 + (left.y - right.y) ** 2) ** 0.5
    if scale < 1e-6:
        return None
    mouth = ((top.x - bottom.x) ** 2 + (top.y - bottom.y) ** 2) ** 0.5
    return mouth / scale


def mouth_center_px(face_landmarks, width, height):
    top, bottom = face_landmarks[MOUTH_TOP], face_landmarks[MOUTH_BOTTOM]
    return (
        int((top.x + bottom.x) / 2 * width),
        int((top.y + bottom.y) / 2 * height),
    )


class EdgeTrigger:
    """Fires once when a boolean condition becomes true, then won't fire
    again until it goes false and becomes true again - a one-shot trigger
    rather than a level/hold trigger."""

    def __init__(self, debounce_seconds=0.0):
        self.debounce_seconds = debounce_seconds
        self.active_since = None
        self.fired = False

    def update(self, is_active, now):
        if not is_active:
            self.active_since = None
            self.fired = False
            return False
        if self.active_since is None:
            self.active_since = now
        if not self.fired and now - self.active_since >= self.debounce_seconds:
            self.fired = True
            return True
        return False
