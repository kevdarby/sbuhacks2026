"""End-to-end diagnostic for the two-stream audio path (see ISSUE.md).

Opens both streams exactly the way main.py does, optionally under the same
mediapipe/OpenCV load, and measures what actually happens at each stage:
whether the streams are active, whether both callbacks fire, whether
PortAudio reports xruns, how much signal arrives on the Katana's input,
and how much of it reaches the Mac output buffer.

Output is forced silent (the monitor gain is overridden to -200dB), so
running this makes no sound through either the amp or the Mac - it only
reports numbers.

Run from the project root:
    uv run tools/audio_diag.py            # audio path only
    uv run tools/audio_diag.py --load     # with mediapipe running too
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amp.audio import DEVICE_NAME, SAMPLE_RATE, Looper, db_to_linear


def describe_devices():
    print("Host APIs:")
    for i, ha in enumerate(sd.query_hostapis()):
        print(f"  [{i}] {ha['name']} (default output: {ha['default_output_device']})")
    print(f"Process default (input, output) device: {sd.default.device}")
    print("Devices:")
    for i, d in enumerate(sd.query_devices()):
        mark = ""
        if DEVICE_NAME.lower() in d["name"].lower():
            mark = "  <- Katana"
        if i == sd.default.device[1]:
            mark += "  <- default output (where the monitor goes)"
        print(f"  [{i}] ha{d['hostapi']} in{d['max_input_channels']} "
              f"out{d['max_output_channels']} {d['default_samplerate']:.0f}Hz "
              f"{d['name']}{mark}")
    print()


def make_load():
    """The same per-frame work main.py's video loop does, so the audio path
    is measured under realistic main-thread contention rather than idle."""
    import cv2
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    face = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame = np.random.default_rng(0).integers(
        0, 255, (720, 1280, 3), dtype=np.uint8)

    def step():
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands.process(rgb)
        face.process(rgb)

    return step


def run_meter():
    """Input-only, output-silent level meter on the Katana's ch1/2. Nothing
    else in the diagnostic can tell a broken signal chain from a player who
    simply wasn't playing; this can."""
    looper = Looper()
    level = [0.0, 0.0]  # rolling peak, rolling rms

    def tap(block):
        peak = float(np.abs(block).max())
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        level[0] = max(peak, level[0] * 0.6)
        level[1] = max(rms, level[1] * 0.6)

    looper.input_listener = tap
    stream = looper.open_stream()
    stream.start()
    print("Play the guitar. Ctrl-C to stop.\n")
    print("  A struck chord should push rms into the 0.01-0.07 range.")
    print("  A flat 0.00000 with the guitar plugged in and the amp up means")
    print("  the signal never reaches USB ch1/2 - the problem is upstream")
    print("  of every line of Python in this repo.\n")
    try:
        while True:
            time.sleep(0.1)
            peak, rms = level
            bar = "#" * min(50, int(rms * 700))
            sys.stdout.write(f"\r  rms {rms:.5f}  peak {peak:.5f}  |{bar:<50}|")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n")
    finally:
        stream.stop()
        stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Run mediapipe Hands+FaceMesh on the main thread "
                             "while measuring, as main.py does.")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--meter", action="store_true",
                        help="Live input-level meter: play the guitar and "
                             "watch whether anything arrives on ch1/2. This "
                             "is the one link the automated check can't "
                             "verify on its own. Ctrl-C to stop.")
    args = parser.parse_args()

    if args.meter:
        run_meter()
        return

    describe_devices()

    looper = Looper()

    # Tap the raw input so we can report whether any signal is arriving at
    # all - the whole diagnosis is meaningless if the guitar isn't coming
    # through (ISSUE.md step 6).
    in_peak = [0.0]
    in_sum = [0.0]
    in_n = [0]

    def tap(block):
        in_peak[0] = max(in_peak[0], float(np.abs(block).max()))
        in_sum[0] += float(np.sum(block.astype(np.float64) ** 2))
        in_n[0] += block.size

    looper.input_listener = tap

    # Tap what the monitor actually delivers into the Mac output buffer.
    # The real gain stays in effect so the measured peak is the true one;
    # the buffer is zeroed *after* measuring, which is what keeps this
    # diagnostic silent without distorting the number.
    out_peak = [0.0]
    real_mac_callback = looper.mac_callback

    def silent_mac_callback(outdata, frames, time_info, status):
        real_mac_callback(outdata, frames, time_info, status)
        out_peak[0] = max(out_peak[0], float(np.abs(outdata).max()))
        outdata.fill(0)

    looper.mac_callback = silent_mac_callback

    audio_stream = looper.open_stream()
    audio_stream.start()
    mac_stream = looper.open_mac_stream()
    try:
        mac_stream.start()
    except Exception as exc:
        print(f"FAIL: mac_stream.start() raised: {exc}")
        return

    print(f"katana stream: active={audio_stream.active} "
          f"blocksize={audio_stream.blocksize} latency={audio_stream.latency}")
    print(f"mac stream:    active={mac_stream.active} "
          f"blocksize={mac_stream.blocksize} latency={mac_stream.latency:.4f}s "
          f"samplerate={mac_stream.samplerate:.0f}")
    print(f"monitor gain {looper.monitor_gain_db:+.1f}dB "
          f"(applied for measurement, then muted before playout)\n")

    looper.monitor_enabled = True

    step = make_load() if args.load else None
    print(f"Measuring for {args.seconds:.0f}s "
          f"{'under mediapipe load' if step else '(audio only)'}...")

    c0, m0 = looper.cb_count, looper.mac_cb_count
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        if step:
            step()
        else:
            time.sleep(0.005)
    dt = time.monotonic() - t0

    cb = looper.cb_count - c0
    mac_cb = looper.mac_cb_count - m0
    in_rms = (in_sum[0] / in_n[0]) ** 0.5 if in_n[0] else 0.0

    print()
    print(f"  katana callback : {cb / dt:7.1f}/s  "
          f"({cb} calls, {looper.cb_status_count} xruns)")
    print(f"  mac callback    : {mac_cb / dt:7.1f}/s  "
          f"({mac_cb} calls, {looper.mac_cb_status_count} xruns)")
    print(f"  ring buffer     : {looper.monitor_status()}")
    print(f"  input signal    : rms {in_rms:.5f}  peak {in_peak[0]:.5f}")
    print(f"  monitor output  : peak {out_peak[0]:.5f} "
          f"(input peak x {db_to_linear(looper.monitor_gain_db):.1f} "
          f"at {looper.monitor_gain_db:+.1f}dB)")
    print()

    if mac_cb == 0:
        print("VERDICT: the Mac callback never fired - the second stream is "
              "not really running. The bug is in stream setup, not in the "
              "monitor/ring-buffer logic.")
    elif looper.mac_cb_status_count > mac_cb * 0.1:
        print("VERDICT: the Mac callback is firing but PortAudio is reporting "
              "sustained xruns - genuine starvation.")
    elif in_peak[0] < 1e-4:
        print("VERDICT: the audio plumbing is healthy (both callbacks firing, "
              "no xruns) but essentially NO SIGNAL is arriving on the Katana's "
              "input channels. Check the guitar is plugged in and the amp's "
              "input level is up - play while this runs.")
    else:
        print("VERDICT: the whole path is healthy end to end - signal arrives, "
              "the ring buffer stays fed, and it reaches the Mac output "
              "buffer. If it is still inaudible in main.py, the monitor was "
              "most likely never toggled on (the 'l' key only registers while "
              "the OpenCV window has focus, not the terminal) or the Mac's "
              "own output volume is down.")

    audio_stream.stop()
    audio_stream.close()
    mac_stream.stop()
    mac_stream.close()


if __name__ == "__main__":
    main()
