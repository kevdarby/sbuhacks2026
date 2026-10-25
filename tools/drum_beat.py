"""Play a drum groove on the solenoid through the USB relay, at a fixed
tempo (see drums/patterns.py for the grooves). For drums that follow the
guitar instead, see tools/drum_follow.py.

Run from the project root: uv run tools/drum_beat.py
Options: --style rock|basic|funk|halftime|gallop|machine, --bpm 90, --swing 0-0.3,
--no-fill (skip the fill every 4th bar), --port PATH. Ctrl+C to stop.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drums.patterns import PULSES, STYLES
from drums.relay import RELAY_RECOVERY, Relay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=STYLES, default="rock")
    parser.add_argument("--bpm", type=float, default=90)
    parser.add_argument("--swing", type=float, default=0.0)
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--port", default=None)
    parser.add_argument("--relay-channel", type=int, default=None,
                        help="Which relay channel the solenoid is on (default 4)")
    args = parser.parse_args()

    groove, fill = STYLES[args.style]
    step = 60.0 / args.bpm / 4
    swing = max(0.0, min(args.swing, 0.3))
    # Swing pulls the gap after each offbeat 16th in; that's the tightest spacing
    tightest = step * (1 - swing)
    longest = max(PULSES[s] for s in groove + fill if s in PULSES)
    if longest + RELAY_RECOVERY > tightest:
        raise SystemExit(f"{args.bpm:g} BPM is too fast for the relay with {args.style} - "
                         "try a lower --bpm, less --swing, or --style gallop/machine")

    relay = Relay(args.port, args.relay_channel)
    print(f"Playing {args.style} at {args.bpm:g} BPM on {relay.port}. Ctrl+C to stop.")
    # Schedule every step from a fixed start time so timing doesn't drift
    start = time.perf_counter()
    n = 0
    try:
        while True:
            bar, pos = divmod(n, 16)
            pattern = fill if not args.no_fill and bar % 4 == 3 else groove
            onset = start + n * step + (swing * step if pos % 2 else 0)
            delay = onset - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            pulse = PULSES.get(pattern[pos])
            if pulse is not None:
                relay.hit(pulse)
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        relay.close()
        print(f"\nStopped after {n // 16} bars.")


if __name__ == "__main__":
    main()
