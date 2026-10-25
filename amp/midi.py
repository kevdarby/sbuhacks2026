"""Roland/Boss sysex protocol for the Katana MkII, confirmed against real
hardware. See HANDOFF.md for how these values were found and what not to
re-try.
"""

import mido

# Model ID is 4 bytes on this amp (confirmed via a known-working community
# script, not the 3-byte Gen-1 doc). Address found empirically: byte 0x28
# in the patch's temporary block (60 00 00 00) changed 60->100 when the
# volume slider was dragged in BOSS Tone Studio, and reading it back via
# RQ1 confirms the value sticks.
MODEL_ID = [0x00, 0x00, 0x00, 0x33]
VOLUME_ADDRESS = [0x60, 0x00, 0x00, 0x28]
DEVICE_ID = 0x00


def roland_checksum(vals):
    accum = 0
    for v in vals:
        accum = (accum + v) & 0x7F
    return (128 - accum) & 0x7F


def build_volume_message(percent):
    value = max(0, min(100, round(percent)))
    payload = MODEL_ID + [0x12] + VOLUME_ADDRESS + [value]
    checksum = roland_checksum(VOLUME_ADDRESS + [value])
    return mido.Message(
        "sysex", data=[0x41, DEVICE_ID] + payload + [checksum]
    )


def find_katana_output():
    for name in mido.get_output_names():
        if "katana" in name.lower():
            return name
    return None


def find_katana_input():
    for name in mido.get_input_names():
        if "katana" in name.lower():
            return name
    return None
