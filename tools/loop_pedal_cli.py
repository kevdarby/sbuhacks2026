"""Standalone keyboard-driven looper test - no camera/mediapipe needed.
Useful for testing the audio path (channel mapping, gain, output routing)
in isolation from the gesture-detection stack.

Run from the project root: uv run tools/loop_pedal_cli.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amp.audio import GAIN_STEP_DB, DEVICE_NAME, SAMPLE_RATE, Looper


def main():
    looper = Looper()

    print(f"Opening {DEVICE_NAME} @ {SAMPLE_RATE}Hz (in: ch1/2, out: ch3/4, "
          f"amp gain {looper.gain_db:+.1f}dB, mac gain {looper.mac_gain_db:+.1f}dB)")
    with looper.open_stream(), looper.open_mac_stream():
        print("\nCommands: 'r' = start/stop recording (stop starts loop "
              "playback), 's' = stop & clear loop, '+'/'-' = adjust "
              "playback level, 't' = toggle loop output between amp/mac, "
              "'l' = toggle live guitar monitor through Mac, "
              "'d' = audio-thread diagnostics, 'q' = quit.\n")
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "r":
                looper.toggle_record()
            elif cmd == "s":
                looper.stop()
            elif cmd == "+":
                target, db = looper.adjust_gain(GAIN_STEP_DB)
                print(f"{target} gain: {db:+.1f}dB")
            elif cmd == "-":
                target, db = looper.adjust_gain(-GAIN_STEP_DB)
                print(f"{target} gain: {db:+.1f}dB")
            elif cmd == "t":
                target = looper.toggle_output_target()
                print(f"Loop output now routed to: {target}")
            elif cmd == "l":
                enabled = looper.toggle_monitor()
                print(f"Live guitar monitor through Mac: {'ON' if enabled else 'off'}")
            elif cmd == "d":
                # Same readout main.py puts on the HUD - this tool is the
                # control case, so comparing the two numbers side by side
                # is how you tell a main.py-specific regression from a
                # problem in the audio path itself.
                print(looper.monitor_status())
            elif cmd == "q":
                break


if __name__ == "__main__":
    main()
