"""Record a take for tuning the drum follower offline: the guitar straight
from the Katana's USB input (processed, plus the dry signal), the room through the Mac's mic (amp and drums
together), and a log of what the tempo tracker thought and when the drums
fired - all timestamped on one clock (time.perf_counter).

Two modes:
  --click BPM   The solenoid plays a steady beat at a fixed tempo and you
                strum along to it. Gives takes where the right answer is
                known, so the tracker can be scored against it.
  (default)     The real thing: tracker + follower, exactly like
                tools/drum_follow.py, so failures can be replayed. With
                --style auto the arranger writes the groove, and what it
                heard and chose is logged for every bar.

Run from the project root:
  uv run tools/record_take.py --click 100 --name click100
  uv run tools/record_take.py --name free1
  uv run tools/record_take.py --style auto --name song1
  uv run tools/record_take.py --style auto --silent --name song1_quiet
  uv run tools/record_take.py --style auto --bass --name song1_bass
Saves to takes/<name>/ (gitignored). Ctrl+C to stop recording.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amp.audio import DRY_ANALYSIS_GAIN, SAMPLE_RATE, Looper
from amp.bass import BASS_STYLES, BassLine, BassSynth
from amp.chords import ChordDetector, parse_chords
from amp import notes
from amp.tempo import CAPTURE_DELAY, TempoTracker
from drums.arranger import Arranger
from drums.follower import DEFAULT_LEAD, BeatFollower, _sleep_until
from drums.patterns import STYLES
from drums.relay import Relay
from paths import PROJECT_ROOT

MIC_DEVICE = "MacBook Air Microphone"
MIC_RATE = 48000
CLICK_PULSE = 0.03


class BlockRecorder:
    """Collects audio blocks with the perf_counter time each block's first
    sample was captured. Appending is all the audio thread does."""

    def __init__(self):
        self.blocks = []

    def add(self, first_sample_time, mono):
        self.blocks.append((first_sample_time, mono.copy()))

    def save(self, path, rate):
        # The streams are still running while this saves, so work from a
        # snapshot rather than a list the callbacks keep appending to
        blocks = list(self.blocks)
        if not blocks:
            return None
        audio = np.concatenate([b for _, b in blocks]).astype(np.float32)
        times = np.array([t for t, _ in blocks])
        lengths = np.array([len(b) for _, b in blocks])
        np.savez(path.with_suffix(".npz"), audio=audio, block_times=times,
                 block_lengths=lengths, rate=rate)
        # A listenable copy too. The raw guitar signal is quiet, so normalize.
        peak = float(np.abs(audio).max()) or 1.0
        with wave.open(str(path.with_suffix(".wav")), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((audio / peak * 0.9 * 32767).astype(np.int16).tobytes())
        return len(audio) / rate


class LoggingArranger(Arranger):
    """Logs what the arranger heard and chose at every bar."""

    def __init__(self, tracker, fallback):
        super().__init__(tracker, fallback)
        self.log = []

    def next_bar(self, bar_start, step_seconds):
        pattern, shift = super().next_bar(bar_start, step_seconds)
        self.log.append({
            "bar_start": bar_start, "step_seconds": step_seconds, "shift": shift,
            "profile": None if self.profile is None else [round(float(v), 3) for v in self.profile],
            "pattern": pattern, "description": self.description,
        })
        return pattern, shift


class LoggingBassLine(BassLine):
    """Logs every bass decision: the chord, whether it came from the
    learned progression, and the note actually scheduled."""

    def __init__(self, synth, chords, **kwargs):
        super().__init__(synth, chords, **kwargs)
        self.log = []
        schedule = synth.schedule

        def logged_schedule(start, duration, freq, velocity=1.0):
            self.log[-1].update(note_freq=freq, duration=duration, velocity=velocity)
            schedule(start, duration, freq, velocity)

        synth.schedule = logged_schedule

    def plan(self, start, duration, beat_seconds, beat_index, pos):
        self.log.append({"start": start, "beat_index": beat_index, "pos": pos,
                         "planned_at": time.perf_counter()})
        super().plan(start, duration, beat_seconds, beat_index, pos)
        self.log[-1].update(chord=self.chord, cycle=self.cycle)

    def history(self):
        return [{"beat_index": j, "start": self._beat_times.get(j), "chord": c}
                for j, c in sorted(self._history.items())]


class LoggingRelay:
    """Passes hits through to the real relay and logs when each was sent.
    With relay=None nothing is switched at all - unplugging the 12V isn't
    enough for a quiet take, since the relay board clicks on its own."""

    def __init__(self, relay):
        self.relay = relay
        self.port = relay.port if relay is not None else None
        self.hits = []

    def hit(self, pulse):
        self.hits.append((time.perf_counter(), pulse))
        if self.relay is not None:
            self.relay.hit(pulse)
        else:
            time.sleep(pulse)  # same timing as a real hit, so the log matches

    def close(self):
        if self.relay is not None:
            self.relay.close()


def play_click(relay, bpm, stop, lead, beats):
    """Steady quarter notes, fired `lead` early so each hit sounds on the
    grid. Logs the grid times (when the hits should be heard)."""
    period = 60 / bpm
    start = time.perf_counter() + 1.0
    n = 0
    while not stop.is_set():
        sound_time = start + n * period
        _sleep_until(sound_time - lead)
        if stop.is_set():
            break
        beats.append(sound_time)
        relay.hit(CLICK_PULSE)
        n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--click", type=float, default=None, metavar="BPM")
    parser.add_argument("--style", choices=["auto", *STYLES], default="basic")
    parser.add_argument("--lead-ms", type=float, default=DEFAULT_LEAD * 1000)
    parser.add_argument("--port", default=None)
    parser.add_argument("--relay-channel", type=int, default=None,
                        help="Which relay channel the solenoid is on (default 4)")
    parser.add_argument("--silent", action="store_true",
                        help="Never switch the relay; drum hits are only logged")
    parser.add_argument("--bass", action="store_true")
    parser.add_argument("--bass-db", type=float, default=0.0)
    parser.add_argument("--bass-octave", type=int, default=-1)
    parser.add_argument("--bass-style", choices=BASS_STYLES, default="drive")
    parser.add_argument("--bass-note", default=None,
                        help="Hold this note under every chord, e.g. E (implies --bass-style pedal)")
    parser.add_argument("--chord-model", choices=["basic-pitch", "spectrum"], default="basic-pitch",
                        help="How the bass reads chords: Basic Pitch on the dry signal (default) or the spectrum")
    parser.add_argument("--chords", default=None,
                        help="The song's chords, e.g. \"Em G D A C\": the bass only picks among these")
    args = parser.parse_args()
    allowed = parse_chords(args.chords) if args.chords else None
    pedal_note = None
    if args.bass_note:
        pedal_note = next(iter(parse_chords(args.bass_note)))[0]
        args.bass_style = "pedal"
    if args.silent and args.click:
        raise SystemExit("--click needs the drum to be heard - drop --silent")

    out_dir = PROJECT_ROOT / "takes" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists - pick another --name")
    lead = args.lead_ms / 1000

    guitar = BlockRecorder()
    guitar_dry = BlockRecorder()
    mic = BlockRecorder()
    looper = Looper()
    tracker = TempoTracker(SAMPLE_RATE)
    chords = None
    if args.bass:
        use_model = args.chord_model == "basic-pitch" and notes.available()
        chords = ChordDetector(SAMPLE_RATE, tracker, allowed=allowed,
                               note_model=notes.NoteModel() if use_model else None)
    bass = None
    if args.bass:
        synth = BassSynth(SAMPLE_RATE, level_db=args.bass_db)
        bass = LoggingBassLine(synth, chords, octave=args.bass_octave, style=args.bass_style,
                                pedal_note=pedal_note)
        looper.output_source = synth.render

    def on_guitar(stereo_block):
        now = time.perf_counter()
        first = now - CAPTURE_DELAY - (len(stereo_block) - 1) / SAMPLE_RATE
        guitar.add(first, stereo_block.mean(axis=1))

    looper.input_listener = on_guitar

    def on_dry(stereo_block):
        now = time.perf_counter()
        first = now - CAPTURE_DELAY - (len(stereo_block) - 1) / SAMPLE_RATE
        guitar_dry.add(first, stereo_block.mean(axis=1))
        # Analysis listens to the dry channels: the processed ones have the
        # bass mixed back in - see DRY_ANALYSIS_GAIN
        scaled = stereo_block * DRY_ANALYSIS_GAIN
        tracker.feed(scaled, now=now)
        if chords is not None:
            chords.feed(scaled)
            chords.feed_dry(stereo_block)

    looper.dry_listener = on_dry

    def on_mic(indata, frames, time_info, status):
        # Map the stream's own capture timestamp onto perf_counter
        offset = time.perf_counter() - time_info.currentTime
        mic.add(time_info.inputBufferAdcTime + offset, indata[:, 0])

    relay = LoggingRelay(None if args.silent else Relay(args.port, args.relay_channel))
    follower = None
    arranger = None
    click_beats = []
    stop = threading.Event()
    estimates = []
    started = time.perf_counter()

    if args.click:
        mode = f"click at {args.click:g} BPM - strum along to the drum"
    else:
        style = args.style
        if style == "auto":
            style = "basic"
            arranger = LoggingArranger(tracker, STYLES[style][0])
        follower = BeatFollower(tracker, relay, style=style, lead=lead, arranger=arranger,
                                bass=bass)
        mode = f"follow mode ({args.style}) - play and let the drums join in"
        if bass is not None:
            mode += f", bass ({args.bass_style}, {looper.gain_db + args.bass_db:+.0f}dB)"
    print(f"Recording {args.name}: {mode}. Ctrl+C to stop.")

    # Streams are started by hand and deliberately never stopped: stopping
    # the Katana and mic streams together deadlocked inside PortAudio
    # (FinishStoppingStream vs CoreAudio's startStopCallback, each waiting
    # on the other's mutex) and lost a whole take that was only held in
    # memory. So the take is saved first and the process exits without
    # closing them - the OS releases the devices.
    streams = [looper.open_stream(latency="low"),
               sd.InputStream(device=MIC_DEVICE, samplerate=MIC_RATE,
                              channels=1, dtype="float32", callback=on_mic)]
    try:
        for stream in streams:
            stream.start()
        tracker.start()
        if follower is not None:
            follower.start()
        else:
            threading.Thread(target=play_click, daemon=True,
                             args=(relay, args.click, stop, lead, click_beats)).start()
        last = None
        while True:
            est = tracker.estimate
            if est is not None and est is not last:
                last = est
                estimates.append({"time": time.perf_counter(), "bpm": est[0],
                                  "beat_time": est[1], "confidence": est[2]})
            line = f"{time.perf_counter() - started:5.1f}s  "
            line += "listening..." if est is None else f"heard {est[0]:5.1f} BPM"
            print(f"\r{line:<40}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    # More Ctrl+Cs while saving would only interrupt the save
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("\nSaving...")
    stop.set()
    if follower is not None:
        follower.stop()
    tracker.stop()
    relay.close()

    out_dir.mkdir(parents=True)
    guitar_secs = guitar.save(out_dir / "guitar", SAMPLE_RATE)
    guitar_dry.save(out_dir / "guitar_dry", SAMPLE_RATE)
    mic_secs = mic.save(out_dir / "mic", MIC_RATE)
    (out_dir / "take.json").write_text(json.dumps({
        "mode": "click" if args.click else "follow",
        "click_bpm": args.click,
        "style": args.style,
        "lead": lead,
        "capture_delay": CAPTURE_DELAY,
        "started": started,
        "relay_hits": [{"time": t, "pulse": p} for t, p in relay.hits],
        "click_beats": click_beats,
        "estimates": estimates,
        "arranger_bars": [] if arranger is None else arranger.log,
        "bass": None if bass is None else {
            "style": args.bass_style, "octave": args.bass_octave, "level_db": args.bass_db,
            "chords": args.chords, "chord_model": args.chord_model,
            "notes": bass.log, "beat_chords": bass.history()},
    }, indent=1))
    print(f"Saved {out_dir.relative_to(PROJECT_ROOT)}: guitar {guitar_secs or 0:.0f}s, "
          f"mic {mic_secs or 0:.0f}s, {len(relay.hits)} drum hits, {len(estimates)} tempo estimates.")
    sys.stdout.flush()
    os._exit(0)  # skip closing the streams - see above


if __name__ == "__main__":
    main()
