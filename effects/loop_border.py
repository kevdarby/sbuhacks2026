import math

from amp.audio import PLAYING, RECORDING

RECORDING_COLOR = (40, 40, 255)   # BGR red
PLAYING_COLOR = (120, 255, 90)    # BGR neon green


def loop_light(state, now):
    """The loop pedal's state as light from the top of the frame (see
    StageLights.top): a fast, urgent red pulse while recording, a slow green
    breathing while the loop plays, nothing when idle. (colour, level) or
    None."""
    if state == RECORDING:
        pulse = (math.sin(now * 7.0) + 1) / 2
        return RECORDING_COLOR, 0.45 + 0.45 * pulse
    if state == PLAYING:
        pulse = (math.sin(now * 2.0) + 1) / 2
        return PLAYING_COLOR, 0.3 + 0.2 * pulse
    return None
