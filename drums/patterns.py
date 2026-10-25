"""Drum grooves for the single solenoid. With one sound, the groove comes
from rhythm and dynamics: each 16th-note step is an accent, a normal hit, a
soft ghost note or a rest, set by how long the relay stays closed.
"""

# Pulse length per symbol. '.' is a rest. Meant as loud/medium/soft, but
# measured with the MacBook mic, every pulse from 10ms to 80ms hits this
# solenoid equally hard - so on the current rig all three sound the same,
# and 'g' ghost notes are full hits.
PULSES = {"X": 0.08, "x": 0.05, "g": 0.025}

# (groove, fill) - one bar each, 16 steps (4 beats x 4 sixteenths).
# The fast styles skip full-strength 'X' hits: the long pulse is what
# limits how fast the relay can go, so they can run up to ~150 BPM.
FILL = "X.x.x.x.xxxxXXXX"
FAST_FILL = "x.x.x.x.xxxxxxxx"
STYLES = {
    # No fill: the burst of 16ths every 4th bar was too much for a style
    # meant to be just a steady beat
    "basic":    ("X...x...X...x...", "X...x...X...x..."),
    "rock":     ("X.g.x..gX.gXx.g.", FILL),
    "funk":     ("X.gxg.xgX.gxg.Xg", FILL),
    "halftime": ("X..g..g.x..g.g..", FILL),
    "gallop":   ("x.ggx.ggx.ggx.gx", FAST_FILL),
    "machine":  ("xgggxggxxgggxgxg", FAST_FILL),
}
