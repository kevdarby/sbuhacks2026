"""Audible tone ladder for the Mac output path (see ISSUE.md).

Plays a test tone through progressively more of the real monitor path, so
that whichever stage you stop hearing pinpoints where the signal dies:

  A  440Hz  raw sd.OutputStream, its own callback        - is the Mac's
                                                           default output
                                                           audible at all?
  B  554Hz  Looper.open_mac_stream() + the ring buffer   - does the real
                                                           second stream and
                                                           monitor path work
                                                           on its own?
  C  659Hz  both streams open, tone injected as if it    - does it survive
            were the guitar arriving on ch1/2              the Katana stream
                                                           being open too?
  D  880Hz  same as C, under mediapipe load              - does it survive
                                                           what main.py does?

Stages B-D go through the genuine monitor code path, including the +20dB
monitor gain, so the tone is injected at a realistic pickup level (~0.03)
and comes out around 0.3 rather than clipping.

Nothing is sent to the amp - its output channels stay zeroed throughout.

    uv run tools/tone_test.py
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amp.audio import (
    INPUT_SLICE, MONITOR_PREFILL_SECONDS, SAMPLE_RATE, Looper, db_to_linear,
)

RAW_AMPLITUDE = 0.4     # stage A, straight out - no monitor gain in the way
GUITAR_AMPLITUDE = 0.03  # stages B-D, a realistic pickup level pre-gain

BEEP = 0.25   # seconds of tone per beep
GAP = 0.18    # silence between beeps within a stage
TAIL = 0.8    # silence at the end of a stage
# Silence before the first beep of every stage. CoreAudio takes a few
# hundred ms after .start() to wake the output device and prime its
# buffers, and anything played inside that window is simply lost. Without
# this lead-in the first stage's opening beep vanishes while later stages
# (which have more beeps, and a ring-buffer prefill ahead of them) survive
# it - which reads as "the simplest case is the broken one" and sends the
# investigation in exactly the wrong direction.
LEAD_IN = 0.6


class Tone:
    """Emits stage N as N distinct beeps, so the stages can be told apart by
    counting them - this matters because the console output does not
    reliably reach the person listening, leaving the ear as the only
    channel. Pitch rises per stage too, as a secondary cue.

    The whole pattern is precomputed and handed out in slices, so callbacks
    just copy from it and the timing is exact regardless of block size."""

    def __init__(self, freq, amplitude, beeps):
        n_beep = int(BEEP * SAMPLE_RATE)
        n_gap = int(GAP * SAMPLE_RATE)
        ramp = int(0.008 * SAMPLE_RATE)  # click-free attack/release

        env = np.ones(n_beep, dtype=np.float32)
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
        t = np.arange(n_beep) / SAMPLE_RATE
        pulse = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32) * env

        parts = [np.zeros(int(LEAD_IN * SAMPLE_RATE), dtype=np.float32)]
        for _ in range(beeps):
            parts.append(pulse)
            parts.append(np.zeros(n_gap, dtype=np.float32))
        parts.append(np.zeros(int(TAIL * SAMPLE_RATE), dtype=np.float32))
        mono = np.concatenate(parts)
        self.wave = np.column_stack([mono, mono])
        self.pos = 0

    @property
    def duration(self):
        return len(self.wave) / SAMPLE_RATE

    def block(self, frames):
        out = np.zeros((frames, 2), dtype=np.float32)
        n = min(frames, len(self.wave) - self.pos)
        if n > 0:
            out[:n] = self.wave[self.pos:self.pos + n]
            self.pos += n
        return out


def announce(stage, beeps, what, freq):
    print(f"\n  [{stage}] {beeps} beep(s) @ {freq:.0f}Hz - {what}")
    print("      playing...", end="", flush=True)


def report(peak):
    print(f" done (peak written to output buffer: {peak:.3f})")


def stage_a():
    announce("A", 1, "raw OutputStream, direct callback", 440)
    tone = Tone(440, RAW_AMPLITUDE, beeps=1)
    peak = [0.0]

    def cb(outdata, frames, time_info, status):
        outdata[:] = tone.block(frames)
        peak[0] = max(peak[0], float(np.abs(outdata).max()))

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2,
                         dtype="float32", callback=cb):
        time.sleep(tone.duration)
    report(peak[0])


def _measured_mac(looper, peak):
    """Wrap mac_callback so we can report the true peak reaching output."""
    real = looper.mac_callback

    def wrapped(outdata, frames, time_info, status):
        real(outdata, frames, time_info, status)
        peak[0] = max(peak[0], float(np.abs(outdata).max()))

    return wrapped


def stage_b():
    announce("B", 2, "Looper mac stream + monitor ring buffer (no Katana stream)", 554)
    looper = Looper()
    looper.monitor_enabled = True
    peak = [0.0]
    looper.mac_callback = _measured_mac(looper, peak)
    tone = Tone(554, GUITAR_AMPLITUDE, beeps=2)

    stop = threading.Event()

    def feeder():
        """Stands in for the Katana input callback, pushing tone into the
        ring buffer at roughly real-time rate."""
        chunk = 1024
        period = chunk / SAMPLE_RATE
        nxt = time.monotonic()
        while not stop.is_set():
            with looper.lock:
                looper._monitor_push(tone.block(chunk))
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))

    t = threading.Thread(target=feeder, daemon=True)
    with looper.open_mac_stream():
        t.start()
        time.sleep(tone.duration + MONITOR_PREFILL_SECONDS)
        stop.set()
    t.join(timeout=1.0)
    report(peak[0])
    return looper


def stage_cd(with_load):
    stage = "D" if with_load else "C"
    freq = 880 if with_load else 659
    beeps = 4 if with_load else 3
    announce(stage, beeps, "both streams open, tone injected as the guitar signal"
             + (" + mediapipe load" if with_load else ""), freq)

    looper = Looper()
    looper.monitor_enabled = True
    peak = [0.0]
    looper.mac_callback = _measured_mac(looper, peak)
    tone = Tone(freq, GUITAR_AMPLITUDE, beeps=beeps)

    # Substitute the tone for whatever the Katana is actually sending, so
    # the full real path runs: Katana callback -> ring buffer -> mac
    # callback, with only the physical guitar replaced.
    real_cb = looper.callback

    def injected(indata, outdata, frames, time_info, status):
        fake = np.array(indata, copy=True)
        fake[:, INPUT_SLICE] = tone.block(frames)
        real_cb(fake, outdata, frames, time_info, status)

    looper.callback = injected

    step = None
    if with_load:
        import cv2
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.6,
            min_tracking_confidence=0.6)
        face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        frame = np.random.default_rng(0).integers(
            0, 255, (720, 1280, 3), dtype=np.uint8)

        def step():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands.process(rgb)
            face.process(rgb)

    with looper.open_stream(), looper.open_mac_stream():
        t0 = time.monotonic()
        limit = tone.duration + MONITOR_PREFILL_SECONDS + 0.3
        while time.monotonic() - t0 < limit:
            if step:
                step()
            else:
                time.sleep(0.02)
    report(peak[0])
    return looper


def main():
    print(__doc__.split("\n\n")[0])
    out_dev = sd.default.device[1]
    print(f"\nMac output device: [{out_dev}] "
          f"{sd.query_devices(out_dev)['name']}")
    print(f"Monitor gain: {Looper().monitor_gain_db:+.1f}dB "
          f"(x{db_to_linear(Looper().monitor_gain_db):.1f})")
    print("\nTurn your Mac's system volume up.")
    print("Four stages play in order. COUNT THE BEEPS - stage 1 is a single")
    print("beep, stage 2 is two, stage 3 is three, stage 4 is four. The pitch")
    print("rises each stage too. Just note the highest count you hear.\n")

    stage_a()
    time.sleep(1.2)
    stage_b()
    time.sleep(1.2)
    lp_c = stage_cd(with_load=False)
    time.sleep(1.2)
    lp_d = stage_cd(with_load=True)

    print("\n  stage C stream health:", lp_c.monitor_status())
    print("  stage D stream health:", lp_d.monitor_status())
    print("\nHighest beep-count you heard -> where the signal dies:")
    print("  nothing at all -> the Mac's own output/volume, not this code")
    print("  1 only         -> the Looper mac stream or the monitor ring buffer")
    print("  up to 2        -> opening the Katana stream breaks the second one")
    print("  up to 3        -> mediapipe load breaks it")
    print("  all 4          -> the whole path works; the fault is the guitar")
    print("                    signal reaching it, or the 'l' toggle in main.py")


if __name__ == "__main__":
    main()
