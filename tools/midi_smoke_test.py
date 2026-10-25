"""Standalone MIDI link check - not part of the main app. Lists MIDI ports,
finds the Katana, sends a generic Identity Request, prints the reply. Good
first check that the MIDI link is alive at all (works regardless of the
Rx-Exclusive-that-wasn't-the-real-issue / model-ID stuff documented in
HANDOFF.md - Identity Request is a universal message, handled differently
than manufacturer sysex).

Run from the project root: uv run tools/midi_smoke_test.py
"""

import sys
import time
from pathlib import Path

import mido

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("Hello from boss-kata!")

    inputs = mido.get_input_names()
    outputs = mido.get_output_names()

    print("\nMIDI inputs:")
    for name in inputs:
        print(f"  {name}")

    print("\nMIDI outputs:")
    for name in outputs:
        print(f"  {name}")

    katana_in = next((n for n in inputs if "katana" in n.lower()), None)
    katana_out = next((n for n in outputs if "katana" in n.lower()), None)

    if not katana_in or not katana_out:
        print("\nNo Katana MIDI port found. Is the amp connected/powered on "
              "and (on Win/Mac) is the Roland driver installed?")
        return

    print(f"\nFound Katana ports: in='{katana_in}' out='{katana_out}'")

    # Roland "Identity Request" universal sysex - any compliant device should
    # reply with an Identity Reply, which is a cheap way to confirm the link
    # is alive before attempting any Katana-specific address reads/writes.
    identity_request = mido.Message(
        "sysex", data=[0x7E, 0x7F, 0x06, 0x01]
    )

    with mido.open_output(katana_out) as outport, \
            mido.open_input(katana_in) as inport:
        outport.send(identity_request)

        deadline = time.monotonic() + 2.0
        reply = None
        while time.monotonic() < deadline:
            reply = inport.poll()
            if reply is not None:
                break
            time.sleep(0.05)

        if reply is None:
            print("\nNo reply within 2s (sent the request, but got silence).")
        else:
            print(f"\nReceived reply: {reply}")


if __name__ == "__main__":
    main()
