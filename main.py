"""Gesture-controlled Katana with a band that follows you: drums on the
solenoid and a bass line out the amp's speaker, locked to the tempo and
chords of what you play. Right hand -> bass volume, mouth -> loop pedal,
with a neon HUD overlay. See HANDOFF.md for the hardware/protocol details
behind amp/, and the modular layout (gestures/, effects/, amp/, drums/)
this wires together.
"""

import argparse
import os
import signal
import sys
import time
import traceback

import cv2
import mediapipe as mp

from amp import notes
from amp.audio import DRY_ANALYSIS_GAIN, GAIN_STEP_DB, IDLE, PLAYING, SAMPLE_RATE, Looper
from amp.bass import BASS_STYLES, BassLine, BassSynth
from amp.chords import ChordDetector, parse_chords
from amp.tempo import TempoTracker
from amp.tuner import PitchAnalyzer
from drums.arranger import Arranger
from drums.follower import DEFAULT_LEAD, BeatFollower
from drums.loopband import LoopBand
from drums.patterns import STYLES
from drums.relay import Relay
from drums.synth import SOFTWARE_LEAD, DrumSynth, SoftwareRelay
from effects.floor_eq import FloorEQ
from effects.loop_border import loop_light
from effects.stage_lights import StageLights, TappedRelay
from effects.trigger_burst import draw_trigger_burst
from effects.tuner_hud import TunerDisplay
from effects.volume_ring import PulseValue, draw_volume_ring
from gestures.calibration import Calibration
from gestures.hand import PalmFacingGate, hand_openness_ratio, palm_facing_signal
from gestures.mouth import EdgeTrigger, mouth_center_px, mouth_openness_ratio
from gestures.smoothing import Smoother
from display import crop_to_aspect, screen_aspect
from paths import CALIBRATION_FILE, MOUTH_CALIBRATION_FILE

MOUTH_TRIGGER_DEBOUNCE_SECONDS = 0.05
# Hand openness -> bass volume: 50% is the bass's normal (auto-levelled)
# volume, fully open doubles it, a fist mutes it
DEFAULT_BASS_PERCENT = 50

NEON_LANDMARK_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=(255, 255, 0), thickness=2, circle_radius=3,
)
NEON_CONNECTION_SPEC = mp.solutions.drawing_utils.DrawingSpec(
    color=(255, 120, 0), thickness=2,
)


class HandControl:
    """One hand's openness, smoothed, frozen while the palm is turned away
    (so the fretting hand - which shows the camera its back - doesn't move
    anything), plus where to draw its ring."""

    def __init__(self, label):
        self.label = label  # MediaPipe handedness: "Right" or "Left"
        self.smoother = Smoother()
        self.gate = PalmFacingGate()
        self.reset_frame()

    def reset_frame(self):
        self.ratio = None
        self.facing = True
        self.wrist_px = None
        self.ring_radius = 70

    def update(self, landmarks, w, h):
        self.facing = self.gate.update(palm_facing_signal(landmarks, self.label))
        if self.facing:
            ratio = hand_openness_ratio(landmarks)
            if ratio is not None:
                self.ratio = self.smoother.update(ratio)
        else:
            self.ratio = self.smoother.value
        wrist, middle_mcp = landmarks[0], landmarks[9]
        self.wrist_px = (int(wrist.x * w), int(wrist.y * h))
        palm_len = (((wrist.x - middle_mcp.x) * w) ** 2 + ((wrist.y - middle_mcp.y) * h) ** 2) ** 0.5
        self.ring_radius = int(max(55, min(150, palm_len * 2.2)))


def list_cameras(max_index=6):
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                found.append((idx, w, h))
        cap.release()
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera", type=int, default=2,
        help="Camera index to use (see --list-cameras). Defaults to 2, "
             "the real webcam on this machine (index 0 is OBS Virtual Camera).",
    )
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen")
    parser.add_argument(
        "--monitor", action="store_true",
        help="Start with the live guitar monitor already on, instead of "
             "needing the 'l' key. Worth having because 'l' is read through "
             "cv2.waitKey, which only sees keypresses while the video window "
             "has focus - press it with the terminal focused and nothing "
             "happens, with no feedback either way.",
    )
    parser.add_argument("--style", choices=["auto", *STYLES], default="auto",
                        help="Drum groove; auto (default) fits one to what you play")
    parser.add_argument("--no-drums", action="store_true",
                        help="No solenoid: play software drums through the amp instead")
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--bass", type=float, default=None, metavar="PERCENT",
                        help="Fix the bass volume (0-100, 50 = normal) instead of setting it "
                             "with the right hand")
    parser.add_argument("--no-sound", action="store_true",
                        help="Run the band as usual (tempo, chords, lights) but play no "
                             "bass or drums: software drums stand in for the solenoid, "
                             "and neither is mixed into the amp")
    parser.add_argument("--chords", default=None,
                        help="The song's chords, e.g. \"Em G D A C\": the bass only picks among these")
    parser.add_argument("--bass-note", default=None,
                        help="Hold this note under every chord, e.g. E (implies --bass-style pedal)")
    parser.add_argument("--bass-style", choices=BASS_STYLES, default="drive")
    parser.add_argument("--bass-octave", type=int, default=-1)
    parser.add_argument("--chord-model", choices=["basic-pitch", "spectrum"], default="basic-pitch")
    parser.add_argument("--port", default=None, help="USB relay serial port (auto-detected)")
    parser.add_argument("--relay-channel", type=int, default=None,
                        help="Which relay channel the solenoid is on (default 4)")
    args = parser.parse_args()

    if args.list_cameras:
        for idx, w, h in list_cameras():
            print(f"  [{idx}] {w}x{h}")
        return

    looper = Looper()
    lights = StageLights()
    floor_eq = FloorEQ(SAMPLE_RATE)
    pitch_analyzer = PitchAnalyzer(SAMPLE_RATE)
    tuner_display = TunerDisplay()

    # --- the band: tempo tracking, drums, bass ---
    # What lasts for the whole run: the synths, the relay, the note model.
    # What the band has worked out about the playing (tempo, beat, chords,
    # the bass line, the drum arrangement) is built by make_band(), so 'r'
    # can throw it away and start it fresh.
    output_synths = []  # what plays out the amp alongside the guitar
    synth = None
    note_model = None
    pedal_note = None
    if not args.no_bass:
        if args.chord_model == "basic-pitch":
            if notes.available():
                note_model = notes.NoteModel()
            else:
                print("Basic Pitch model or onnxruntime missing - reading chords from the spectrum")
        if args.bass_note:
            pedal_note = next(iter(parse_chords(args.bass_note)))[0]
            args.bass_style = "pedal"
        synth = BassSynth(SAMPLE_RATE)
        synth.master = (DEFAULT_BASS_PERCENT if args.bass is None else args.bass) / 50
        synth.on_note = lights.on_bass_note
        output_synths.append(synth)

    # Analysis listens to the dry channels: the processed ones have the
    # bass (and loop playback) mixed back in - see DRY_ANALYSIS_GAIN
    def on_dry(block):
        scaled = block * DRY_ANALYSIS_GAIN
        pitch_analyzer.feed(scaled)
        floor_eq.feed_guitar(scaled)
        tracker.feed(scaled)
        if chords is not None:
            chords.feed(scaled)
            chords.feed_dry(block)

    looper.dry_listener = on_dry

    relay = None
    if not (args.no_drums or args.no_sound):
        try:
            relay = Relay(args.port, args.relay_channel)
        except (SystemExit, OSError) as exc:
            print(f"No USB relay ({exc}) - playing software drums through the amp")
    lead = DEFAULT_LEAD
    drum_synth = None
    if relay is None:
        drum_synth = DrumSynth(SAMPLE_RATE)
        output_synths.append(drum_synth)
        relay = SoftwareRelay(drum_synth)
        lead = SOFTWARE_LEAD
    relay = TappedRelay(relay, lights.events, lead=lead)

    # The bass and software drums, mixed into the amp's output. With
    # --no-sound they're still rendered, so their queues keep moving, just
    # not played.
    def render_band(frames, time_info):
        blocks = [b for b in (s.render(frames, time_info) for s in output_synths) if b is not None]
        mix = sum(blocks) if blocks else None
        floor_eq.feed_band(mix, frames)  # shown even with --no-sound, like the lights
        if args.no_sound:
            return None
        return mix

    looper.output_source = render_band

    def make_band():
        tracker = TempoTracker(SAMPLE_RATE)
        chords = bass = None
        if synth is not None:
            chords = ChordDetector(SAMPLE_RATE, tracker,
                                   allowed=parse_chords(args.chords) if args.chords else None,
                                   note_model=note_model)
            bass = BassLine(synth, chords, octave=args.bass_octave, style=args.bass_style,
                            pedal_note=pedal_note)
        style = args.style
        arranger = None
        if style == "auto":
            style = "basic"
            arranger = Arranger(tracker, STYLES[style][0])
        follower = BeatFollower(tracker, relay, style=style, arranger=arranger, bass=bass, lead=lead)
        # The loop pedal goes through this: while a loop plays, the band
        # replays what it played while the loop was recorded
        loop_band = LoopBand(looper, follower, relay)
        return tracker, chords, bass, arranger, follower, loop_band

    tracker, chords, bass, arranger, follower, loop_band = make_band()

    # Low latency (7ms blocks): the bass has to be scheduled ahead by the
    # output latency, and the looper and live monitor were checked to run
    # glitch-free this way too
    audio_stream = looper.open_stream(latency="low")
    audio_stream.start()
    tracker.start()
    follower.start()
    # The Mac-output stream is the second, independent stream (see
    # amp/audio.py) and is the one that has historically failed quietly -
    # so don't let a failure to open/start it look like "no sound".
    mac_stream = looper.open_mac_stream()
    try:
        mac_stream.start()
    except Exception as exc:
        print(f"WARNING: could not start Mac output stream: {exc}")
        print("Loop-to-Mac routing and the live monitor will be silent.")
    if not mac_stream.active:
        print("WARNING: Mac output stream is not active after start().")
    if args.monitor:
        looper.toggle_monitor()
        print("Live guitar monitor: ON (--monitor)")

    cap = cv2.VideoCapture(args.camera)
    # Every frame is cropped to the screen's shape, so fullscreen fills the
    # screen instead of leaving a bar - and the side lights and hands are
    # worked out on the same picture that's shown
    aspect = screen_aspect()
    window = "Gesture Volume + Loop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.")
        relay.close()
        os._exit(1)  # audio streams are running; see the note at the end

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils

    # Right hand -> bass volume, left hand -> how busy the drums are. Both
    # use the same fist/open calibration.
    bass_hand = HandControl("Right")
    drum_hand = HandControl("Left")
    vol_calibration = Calibration()
    if vol_calibration.load(CALIBRATION_FILE):
        print(f"Loaded volume calibration: fist={vol_calibration.fist_ratio:.3f} "
              f"open={vol_calibration.open_ratio:.3f}")
    else:
        print("Hands not calibrated: press 'f' with a fist, then 'o' with your hand open")

    mouth_smoother = Smoother(alpha=0.5)  # low latency, this needs to be fast
    mouth_calibration = Calibration()
    if mouth_calibration.load(MOUTH_CALIBRATION_FILE):
        print(f"Loaded mouth calibration: closed={mouth_calibration.fist_ratio:.3f} "
              f"open={mouth_calibration.open_ratio:.3f}")
    else:
        print("Mouth not calibrated: press 'c' with it closed, then 'm' with it open")
    mouth_trigger = EdgeTrigger(MOUTH_TRIGGER_DEBOUNCE_SECONDS)

    bass_percent = DEFAULT_BASS_PERCENT if args.bass is None else args.bass
    exit_code = 0
    vol_pulse = PulseValue(initial=bass_percent)
    drum_pulse = PulseValue()
    drum_percent = None  # None = the guitar's feel decides, until the left hand is used
    last_trigger_time = -999.0
    last_mouth_px = None
    last_frame_time = time.monotonic()
    last_errors = 0
    last_band_phase = None

    if args.bass is not None:
        print(f"Bass volume fixed at {args.bass:g} (--bass); the right hand does nothing")
    print("Right hand = bass volume, left hand = how wild the drums get (fist calm, "
          "open crazy; 'a' hands the drums back to automatic). Press 'f' fist / 'o' open "
          "to calibrate.")
    print("Mouth = loop control (press 'c' mouth closed / 'm' mouth open to "
          "calibrate). Open your mouth (or press spacebar) to cycle: "
          "record -> loop -> stop.")
    print("Press '+'/'-' to adjust playback level (live monitor gain if "
          "it's on, otherwise the loop's), 't' to toggle loop output "
          "between the amp and your Mac's default output.")
    print("Press 'p' to pause/resume the band, 'r' to reset it (and clear the loop).")
    print("Press 'l' to toggle live guitar monitoring through your Mac's "
          "default output (mute the amp's volume yourself for quiet "
          "practice), 'q' to quit.")
    print("NOTE: keys are read by the video window, not this terminal - "
          "click the window first or they do nothing. Or start with "
          "--monitor to skip the toggle entirely.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(crop_to_aspect(frame, aspect), 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            now = time.monotonic()
            dt = max(1e-3, now - last_frame_time)
            last_frame_time = now

            hand_result = hands.process(rgb)
            face_result = face_mesh.process(rgb)

            # --- hands: right = bass volume, left = drum intensity ---
            bass_hand.reset_frame()
            drum_hand.reset_frame()
            for hand_landmarks, handedness in zip(hand_result.multi_hand_landmarks or [],
                                                  hand_result.multi_handedness or []):
                label = handedness.classification[0].label
                control = bass_hand if label == "Right" else drum_hand
                control.update(hand_landmarks.landmark, w, h)
                drawer.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    NEON_LANDMARK_SPEC, NEON_CONNECTION_SPEC,
                )
            vol_ratio = bass_hand.ratio  # for the calibration keys

            # With --bass the volume is fixed: the right hand does nothing
            percent = (vol_calibration.to_percent(bass_hand.ratio)
                       if bass_hand.ratio is not None and bass_hand.facing
                       and args.bass is None else None)
            if percent is not None:
                bass_percent = percent
                if bass is not None:
                    bass.synth.master = bass_percent / 50
                vol_pulse.update(percent, now, dt)
                draw_volume_ring(
                    frame, bass_hand.wrist_px, bass_hand.ring_radius, vol_pulse.value,
                    pop_scale=vol_pulse.pop_scale(now), caption="BASS",
                )

            drum_now = (vol_calibration.to_percent(drum_hand.ratio)
                        if drum_hand.ratio is not None and drum_hand.facing else None)
            if drum_now is not None:
                drum_percent = drum_now
                drum_pulse.update(drum_now, now, dt)
                draw_volume_ring(
                    frame, drum_hand.wrist_px, drum_hand.ring_radius, drum_pulse.value,
                    pop_scale=drum_pulse.pop_scale(now), caption="DRUMS",
                )
            if arranger is not None:
                # Takes effect from the next bar
                arranger.intensity = None if drum_percent is None else drum_percent / 100

            # --- loop control (mouth) ---
            mouth_ratio = None
            mouth_px = last_mouth_px
            if face_result.multi_face_landmarks:
                landmarks = face_result.multi_face_landmarks[0].landmark
                raw = mouth_openness_ratio(landmarks)
                if raw is not None:
                    mouth_ratio = mouth_smoother.update(raw)
                mouth_px = mouth_center_px(landmarks, w, h)
                last_mouth_px = mouth_px

            mouth_open = False
            if mouth_ratio is not None and mouth_calibration.is_ready:
                lo, hi, margin = mouth_calibration.midpoint_threshold()
                mouth_open = mouth_ratio > (lo + hi) / 2 + margin

            if mouth_trigger.update(mouth_open, now):
                loop_band.trigger()
                last_trigger_time = now
            loop_band.update()

            # The LED lights: from the right flaring on each bass note, from
            # the left on each drum hit, and from the top for the loop pedal
            lights.top = loop_light(looper.state, now)
            lights.draw(frame, dt)
            # The spectrum along the bottom, after the lights so they don't
            # tint it
            floor_eq.draw(frame, dt)

            # Audio-thread errors, read from here rather than printed from
            # the callbacks. monitor_status() says which kind of "no sound"
            # it is: mac cb stuck at 0 = the stream isn't running; xruns
            # climbing = PortAudio starvation; fill pinned at 0 with
            # underruns climbing = the Katana side isn't feeding the ring
            # buffer.
            errors = looper.cb_errors + looper.mac_cb_errors
            if errors > last_errors:
                print(f"AUDIO ERROR ({errors}): {looper.last_error}  [{looper.monitor_status()}]")
                last_errors = errors
            if mouth_px is not None:
                draw_trigger_burst(frame, mouth_px, now - last_trigger_time)

            tuner_display.draw(frame, pitch_analyzer.detect())

            # --- band status, to the terminal when it changes ---
            estimate = tracker.estimate
            bpm = follower.bpm  # read once: the drum thread can clear it any time
            if looper.state == PLAYING and follower.paused:
                band = "replaying with the loop"
                if loop_band.loop_bpm is not None:
                    band += f" ({loop_band.loop_bpm:.0f} BPM)"
            elif estimate is None:
                band = "listening..."
            elif not follower.playing or bpm is None:
                band = f"locking on ({estimate[0]:.0f} BPM)"
            else:
                band = f"playing at {bpm:.0f} BPM"
            # Only the state, not the number: the BPM wobbles every frame
            phase = band.split(" (")[0].split(" at ")[0]
            if phase != last_band_phase:
                print(f"Band: {band}")
                last_band_phase = phase

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                # Same trigger the mouth gesture fires - handy when your
                # hands are free but you don't want to open your mouth
                # (or while calibrating/testing without a face in frame).
                loop_band.trigger()
                last_trigger_time = now
            elif key == ord("+") or key == ord("="):
                target, db = looper.adjust_gain(GAIN_STEP_DB)
                print(f"Loop gain ({target}): {db:+.1f}dB")
            elif key == ord("-"):
                target, db = looper.adjust_gain(-GAIN_STEP_DB)
                print(f"Loop gain ({target}): {db:+.1f}dB")
            elif key == ord("t"):
                target = looper.toggle_output_target()
                print(f"Loop output now routed to: {target}")
            elif key == ord("l"):
                enabled = looper.toggle_monitor()
                print(f"Live guitar monitor through Mac: "
                      f"{'ON' if enabled else 'off'}")
            elif key == ord("p"):
                # Pause the band: it keeps following the tempo, so it comes
                # back in time, but plays nothing - and the red and blue
                # lights go out with it
                follower.muted = not follower.muted
                lights.sides = not follower.muted
                if follower.muted:
                    # (not `synth`: that's the bass synth make_band() uses)
                    for s in (bass.synth if bass else None, drum_synth):
                        if s is not None:
                            s.clear()  # notes already queued
                print("Band paused ('p' to resume)" if follower.muted else "Band resumed")
            elif key == ord("r"):
                # Reset the band: it forgets the tempo, beat and chords it
                # had (where it goes wrong when it does) and re-locks onto
                # what's played next. Also stops and clears the loop,
                # unpauses, and hands the drums back to automatic. Volumes,
                # monitor and calibration stay as they are.
                loop_band.stop()
                follower.stop()
                tracker.stop()
                if looper.state != IDLE:
                    looper.stop()
                for s in output_synths:
                    s.clear()  # notes already queued
                # on_dry reads tracker and chords from here, so it feeds
                # the new ones from the next audio block
                tracker, chords, bass, arranger, follower, loop_band = make_band()
                tracker.start()
                follower.start()
                lights.sides = True
                drum_percent = None
                last_band_phase = None
                print("Band reset - play and it'll lock on again")
            elif key == ord("a"):
                drum_percent = None
                print("Drums back to automatic (following the guitar's feel)")
            elif key == ord("f") and vol_ratio is not None:
                vol_calibration.set_fist(vol_ratio)
                vol_calibration.save(CALIBRATION_FILE)
                print(f"Calibrated fist ratio = {vol_ratio:.3f} (saved)")
            elif key == ord("o") and vol_ratio is not None:
                vol_calibration.set_open(vol_ratio)
                vol_calibration.save(CALIBRATION_FILE)
                print(f"Calibrated open ratio = {vol_ratio:.3f} (saved)")
            elif key == ord("c") and mouth_ratio is not None:
                mouth_calibration.set_fist(mouth_ratio)
                mouth_calibration.save(MOUTH_CALIBRATION_FILE)
                print(f"Calibrated mouth-closed ratio = {mouth_ratio:.3f} (saved)")
            elif key == ord("m") and mouth_ratio is not None:
                mouth_calibration.set_open(mouth_ratio)
                mouth_calibration.save(MOUTH_CALIBRATION_FILE)
                print(f"Calibrated mouth-open ratio = {mouth_ratio:.3f} (saved)")
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        loop_band.stop()
        follower.stop()
        tracker.stop()
        relay.close()  # never leave the solenoid energized
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        face_mesh.close()
    # The audio streams are deliberately not stopped: stopping them can
    # deadlock inside PortAudio (see tools/record_take.py), hanging the app
    # on quit. The OS releases the devices when the process ends.
    sys.stdout.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
