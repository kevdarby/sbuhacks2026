"""Fire the solenoid on a steady beat through the USB relay - no amp,
camera or mediapipe needed. Useful for checking the drum rig's wiring,
strike strength and mounting in isolation.

Run from the project root: uv run tools/drum_metronome.py
Options: --bpm 90, --pulse 0.08 (seconds the relay stays closed per hit),
--beats N (stop after N hits; default runs until Ctrl+C), --port PATH.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drums.relay import MAX_PULSE, Relay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpm", type=float, default=90)
    parser.add_argument("--pulse", type=float, default=0.08)
    parser.add_argument("--beats", type=int, default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--relay-channel", type=int, default=None,
                        help="Which relay channel the solenoid is on (default 4)")
    args = parser.parse_args()

    pulse = min(args.pulse, MAX_PULSE)
    interval = 60.0 / args.bpm
    if pulse > interval / 2:
        raise SystemExit(f"{args.bpm:g} BPM is too fast for a {pulse * 1000:.0f}ms pulse")

    relay = Relay(args.port, args.relay_channel)
    print(f"Drumming at {args.bpm:g} BPM on {relay.port} ({pulse * 1000:.0f}ms hits). Ctrl+C to stop.")
    # Schedule each beat from a fixed start time so timing doesn't drift
    start = time.perf_counter()
    beat = 0
    try:
        while args.beats is None or beat < args.beats:
            delay = start + beat * interval - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            relay.hit(pulse)
            beat += 1
    except KeyboardInterrupt:
        pass
    finally:
        relay.close()
        print(f"\nStopped after {beat} hits.")


if __name__ == "__main__":
    main()
