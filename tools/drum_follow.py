"""Play drums along with the guitar: listens to the Katana's USB input,
works out the tempo and where the beats fall, and plays a groove on the
solenoid locked to it. No camera/mediapipe needed.

Run from the project root: uv run tools/drum_follow.py
Options: --style auto|rock|basic|funk|halftime|gallop|machine (auto, the
default, writes a groove to fit what you're playing and rewrites it every
bar - see drums/arranger.py), --no-fill,
--lead-ms 57 (fire the relay this early - raise it if the drums sound
late, lower it if they rush), --listen (track tempo and print it without
playing drums), --port PATH, --bass (play a bass line out the amp's
speaker, following your chords - see amp/bass.py), --bass-db N (bass
level relative to the loop playback gain), --bass-octave N (default -1,
a real bass's range; 0 is an octave higher), --bass-style
drive|eighths|kick|pedal (default drive: eighth notes with octave jumps
and approach notes into chord changes; pedal holds one note - the
guitar's most common - under every chord), --bass-note E (pedal on this
note), --chords "Em G D A C" (the song's chords - the bass then only
picks among these, far more reliably), --chord-model basic-pitch|spectrum
(default basic-pitch: Spotify's note model on the Katana's dry signal - see
amp/notes.py), --no-drums (keep the solenoid quiet, e.g. for bass only).
Ctrl+C to stop.
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amp.audio import DEVICE_NAME, DRY_ANALYSIS_GAIN, SAMPLE_RATE, Looper
from amp.bass import BASS_STYLES, BassLine, BassSynth
from amp.chords import ChordDetector, chord_name, parse_chords
from amp import notes
from amp.tempo import TempoTracker
from drums.arranger import Arranger, chart
from drums.follower import DEFAULT_LEAD, BeatFollower
from drums.patterns import STYLES
from drums.relay import Relay, SilentRelay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", choices=["auto", *STYLES], default="auto")
    parser.add_argument("--no-fill", action="store_true")
    parser.add_argument("--lead-ms", type=float, default=DEFAULT_LEAD * 1000)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--port", default=None)
    parser.add_argument("--relay-channel", type=int, default=None,
                        help="Which relay channel the solenoid is on (default 4)")
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
    parser.add_argument("--no-drums", action="store_true")
    args = parser.parse_args()
    allowed = parse_chords(args.chords) if args.chords else None
    pedal_note = None
    if args.bass_note:
        pedal_note = next(iter(parse_chords(args.bass_note)))[0]
        args.bass_style = "pedal"

    if args.listen:
        relay = None
    else:
        relay = SilentRelay() if args.no_drums else Relay(args.port, args.relay_channel)
    looper = Looper()
    tracker = TempoTracker(SAMPLE_RATE)
    bass = None
    chords = None
    if args.bass and relay is not None:
        note_model = None
        if args.chord_model == "basic-pitch":
            if notes.available():
                note_model = notes.NoteModel()
            else:
                print("Basic Pitch model or onnxruntime missing - reading chords from the spectrum")
        chords = ChordDetector(SAMPLE_RATE, tracker, allowed=allowed, note_model=note_model)
        synth = BassSynth(SAMPLE_RATE, level_db=args.bass_db)
        bass = BassLine(synth, chords, octave=args.bass_octave, style=args.bass_style,
                                pedal_note=pedal_note)
        looper.output_source = synth.render

    # Analysis listens to the dry channels: the processed ones have the
    # bass mixed back in - see DRY_ANALYSIS_GAIN
    def on_dry(block):
        scaled = block * DRY_ANALYSIS_GAIN
        tracker.feed(scaled)
        if chords is not None:
            chords.feed(scaled)
            chords.feed_dry(block)

    looper.dry_listener = on_dry
    follower = None
    arranger = None
    if relay is not None:
        style = args.style
        if style == "auto":
            # Plain beats until there's a bar of guitar to fit a groove to
            style = "basic"
            arranger = Arranger(tracker, STYLES[style][0])
        follower = BeatFollower(tracker, relay, style=style, fill=not args.no_fill,
                                lead=args.lead_ms / 1000, arranger=arranger, bass=bass)

    where = "listening only" if relay is None else f"drums on {relay.port}"
    if bass is not None:
        where += f", bass at {looper.gain_db + args.bass_db:+.0f}dB"
    print(f"Listening on {DEVICE_NAME} ({where}). Start playing - "
          f"it needs a few seconds of steady playing to lock on. Ctrl+C to stop.")
    # Started by hand and never stopped: stopping the Katana stream can
    # deadlock inside PortAudio (see tools/record_take.py), so on exit the
    # relay is switched off and the process ends without closing it
    stream = looper.open_stream(latency="low")
    try:
        stream.start()
        tracker.start()
        if follower is not None:
            follower.start()
        shown = None
        while True:
            if arranger is not None and arranger.profile is not None:
                view = (chart(arranger.profile), arranger.pattern, arranger.description)
                if view != shown:
                    shown = view
                    print(f"\n         1         2         3         4"
                          f"\n  you:   {view[0]}"
                          f"\n  drums: {chart(view[1])}   ({view[2]})")
            estimate = tracker.estimate
            if estimate is None:
                line = "listening..."
            else:
                bpm, _, confidence = estimate
                line = f"heard {bpm:5.1f} BPM (confidence {confidence:.2f})"
                drum_bpm = follower.bpm if follower is not None else None
                if drum_bpm is not None:
                    line += f"  drums at {drum_bpm:5.1f} BPM"
                if bass is not None and bass.chord is not None:
                    line += f"  bass: {chord_name(*bass.chord)}"
                    if bass.chords.source == "basic pitch":
                        line += " (bp)"
            print(f"\r{line:<75}", end="", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if follower is not None:
        follower.stop()
    tracker.stop()
    if relay is not None:
        relay.close()
    print("\nStopped.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
