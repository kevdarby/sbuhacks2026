"""USB audio I/O for the Katana's software looper. See HANDOFF.md for how
the channel mapping and gain were found empirically.
"""

import json
import threading
import time

import numpy as np
import sounddevice as sd

from amp.bass import OUTPUT_EXTRA_DELAY
from paths import GAIN_FILE, MAC_GAIN_FILE, MONITOR_GAIN_FILE

DEVICE_NAME = "KATANA"
SAMPLE_RATE = 44100

# Confirmed empirically: USB input channels 1/2 (index 0/1) carry the live
# guitar/mic signal; output channels 3/4 (index 2/3) are what's actually
# routed to the amp's speaker. Channels 1/2 out and 3/4 in are unused here.
IN_CHANNELS = 4
OUT_CHANNELS = 4
INPUT_SLICE = slice(0, 2)
# Channels 3/4 carry the dry guitar (before the preamp and effects) - Roland's
# documented layout for the MkII on a Mac; ~10dB quieter than 1/2, which is
# why it first looked silent
DRY_SLICE = slice(2, 4)
# Anything that analyzes the playing (tempo, chords, tuner) listens to the
# dry channels, scaled by this to roughly the processed channels' level so
# their thresholds still apply (measured 7.8-10x quieter). Not ch1/2: the
# Katana mixes its USB playback back into them - the bass came back there
# louder than the guitar, 31ms later, and non-linearly (a fitted filter
# cancelled only 4dB), so the drums ended up following the bass. The dry
# channels carried it ~35dB below the playing level.
DRY_ANALYSIS_GAIN = 8.0
OUTPUT_SLICE = slice(2, 4)

# The amp's USB playback path (ch3/4) is calibrated hotter than its
# instrument input path (ch1/2) - it's meant for already-mastered backing
# tracks, not a raw pickup signal - so unity gain plays back much louder
# than it was actually played. There's no way to derive the correct
# attenuation in software; it's tuned by ear once and persisted. The Mac's
# own output (built-in speakers/AirPods/whatever) doesn't have this
# problem - it's a normal consumer output level - so it gets its own,
# separately persisted gain rather than reusing the amp's extreme trim.
DEFAULT_GAIN_DB = -20.0
DEFAULT_MAC_GAIN_DB = 0.0
# The raw captured guitar signal is quiet (a pickup-level signal, not a
# mastered/amplified one - measured RMS around 0.01-0.07 in earlier
# testing) - playing it back at unity through the Mac's speakers is
# inaudible or close to it, so the live monitor needs real boost, not
# just pass-through.
DEFAULT_MONITOR_GAIN_DB = 20.0
GAIN_STEP_DB = 1.0

IDLE, RECORDING, PLAYING = "idle", "recording", "playing"

# Loop playback can go out the amp's speaker (default) or, toggled, out
# whatever the Mac's own current default output device is (built-in
# speakers, AirPods, whatever's selected in System Settings) instead -
# lets you practice the loop pedal without the amp making noise.
OUTPUT_TO_AMP, OUTPUT_TO_MAC = "amp", "mac"

# Live monitor: streams the *live* guitar signal itself (not the recorded
# loop) to the Mac's default output, independent of the loop routing above
# - for quiet practice with the amp's own master volume muted (via the
# hand-gesture volume control) but the guitar still audible through
# speakers/headphones. A ring buffer decouples the two audio streams'
# independent hardware clocks/callback timing (the Katana input and the
# Mac output aren't the same stream, so samples can't be handed off
# synchronously).
#
# The buffer's *size* is only a ceiling, though - it does nothing on its
# own. The reader has to be held back until a cushion has accumulated,
# otherwise it drains the buffer on its very first pull and then rides
# permanently at zero available, where every scheduling hiccup is a
# dropout rather than a glitch. That's what made this silent in main.py
# (mediapipe + the tuner's YIN loop jitter the main thread, and hence the
# callback scheduling, badly) while working fine in the jitter-free
# standalone CLI, and why enlarging the buffer alone never helped.
# PREFILL is the actual latency/robustness knob; BUFFER is just headroom
# above it for the opposite case (writer temporarily outpacing reader).
#
# Sizing: the Katana duplex stream was measured delivering ~93ms input
# blocks (about 10.6 callbacks/sec) while the Mac output pulls ~512-frame
# blocks at ~86/sec, so the reader will routinely drain for most of a
# Katana block before the next one lands. The cushion therefore has to
# exceed one input block; 150ms clears it with margin without adding more
# delay than a live monitor can stand (total path is roughly one Katana
# block + this + the Mac stream's own ~35ms).
MONITOR_BUFFER_SECONDS = 1.0
MONITOR_PREFILL_SECONDS = 0.15


def db_to_linear(db):
    return 10 ** (db / 20)


def load_gain_db(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())["gain_db"]
        except (json.JSONDecodeError, KeyError):
            pass
    return default


def save_gain_db(path, gain_db):
    path.write_text(json.dumps({"gain_db": gain_db}))


class Looper:
    def __init__(self):
        self.state = IDLE
        self.record_chunks = []
        self.loop_buffer = None
        # Timing, on the perf_counter clock, so a band can replay in sync
        # with the loop (see drums/loopband.py): when the first recorded
        # sample was captured, and when the loop's first sample is heard
        # from the amp (set by the first playing block)
        self.record_start_time = None
        self.play_start_time = None
        # Set by finish_recording() to stop at an exact length, once that
        # much has been recorded
        self._stop_samples = None
        self.playhead = 0
        self.lock = threading.Lock()
        self.gain_db = load_gain_db(GAIN_FILE, DEFAULT_GAIN_DB)
        self.mac_gain_db = load_gain_db(MAC_GAIN_FILE, DEFAULT_MAC_GAIN_DB)
        self.monitor_gain_db = load_gain_db(MONITOR_GAIN_FILE, DEFAULT_MONITOR_GAIN_DB)
        # Optional hook: called with each raw input block (stereo, ch1/2) on
        # the audio thread, regardless of record/play state - e.g. for a
        # PitchAnalyzer that wants a live feed independent of looping.
        self.input_listener = None
        # Optional hook: called with (frames, time_info) on the audio thread
        # for extra mono audio to mix into the amp's speaker output (e.g. a
        # BassSynth), at the same tuned playback gain as the loop. Returns
        # None when silent.
        self.output_source = None
        # Optional hook: called with each dry input block (ch3/4) on the
        # audio thread, like input_listener
        self.dry_listener = None
        self.output_target = OUTPUT_TO_AMP

        self.monitor_enabled = False
        monitor_len = int(MONITOR_BUFFER_SECONDS * SAMPLE_RATE)
        self._monitor_buf = np.zeros((monitor_len, 2), dtype=np.float32)
        self._monitor_prefill = int(MONITOR_PREFILL_SECONDS * SAMPLE_RATE)
        self._monitor_write = 0
        self._monitor_read = 0
        self._monitor_available = 0
        # Start (and restart after any underrun) held back until a cushion
        # has built up - see MONITOR_PREFILL_SECONDS.
        self._monitor_priming = True

        # Diagnostics. These are the cheap version of "add a print in the
        # callback": counters bumped on the audio threads and read from the
        # video/UI thread, so nothing does I/O from a realtime callback.
        # monitor_status() formats them.
        self.cb_count = 0
        self.cb_status_count = 0
        self.mac_cb_count = 0
        self.mac_cb_status_count = 0
        self.monitor_underruns = 0
        self.last_status = ""
        self.cb_errors = 0
        self.mac_cb_errors = 0
        self.last_error = ""

    def _clear_monitor_buffer(self):
        """Caller must hold self.lock."""
        self._monitor_write = 0
        self._monitor_read = 0
        self._monitor_available = 0
        self._monitor_priming = True
        self.monitor_underruns = 0

    def toggle_monitor(self):
        with self.lock:
            self.monitor_enabled = not self.monitor_enabled
            # Reset so enabling doesn't play back stale/silent buffer
            # content left over from before it was last turned off.
            self._clear_monitor_buffer()
            return self.monitor_enabled

    def reset_monitor(self):
        """Drop buffered audio without changing whether the monitor is on -
        for when the streams underneath it have been torn down and reopened
        (e.g. following the output device to newly connected AirPods) and
        the samples still in the buffer are stale."""
        with self.lock:
            self._clear_monitor_buffer()

    def monitor_status(self):
        """One-line diagnostic for the HUD/console. Distinguishes the three
        failure modes that all look like 'no sound' from outside: callbacks
        not firing at all (counts stay at 0), PortAudio reporting xruns
        (status counts climbing), and buffer starvation (underruns climbing
        / fill stuck near 0)."""
        with self.lock:
            fill = self._monitor_available / SAMPLE_RATE
            priming = self._monitor_priming
        parts = [f"cb {self.cb_count}/{self.cb_status_count}xrun",
                 f"mac {self.mac_cb_count}/{self.mac_cb_status_count}xrun",
                 f"fill {fill * 1000:.0f}ms{' PRIMING' if priming else ''}",
                 f"under {self.monitor_underruns}"]
        if self.cb_errors or self.mac_cb_errors:
            parts.append(f"ERRORS {self.cb_errors}/{self.mac_cb_errors}")
        if self.last_error:
            parts.append(f"[{self.last_error}]")
        elif self.last_status:
            parts.append(f"[{self.last_status}]")
        return "  ".join(parts)

    def _monitor_push(self, block):
        cap = len(self._monitor_buf)
        n = len(block)
        if n >= cap:
            self._monitor_buf[:] = block[-cap:]
            self._monitor_write = 0
            self._monitor_read = 0
            self._monitor_available = cap
            return
        end = self._monitor_write + n
        if end <= cap:
            self._monitor_buf[self._monitor_write:end] = block
        else:
            first = cap - self._monitor_write
            self._monitor_buf[self._monitor_write:] = block[:first]
            self._monitor_buf[:n - first] = block[first:]
        self._monitor_write = end % cap
        # On overflow the oldest samples have just been overwritten, so the
        # read pointer has to move with them - otherwise it keeps pointing
        # at data that no longer exists and `available` stops describing
        # the real gap between the two pointers.
        overflow = self._monitor_available + n - cap
        if overflow > 0:
            self._monitor_read = (self._monitor_read + overflow) % cap
        self._monitor_available = min(cap, self._monitor_available + n)

    def _monitor_pull(self, frames):
        cap = len(self._monitor_buf)
        out = np.zeros((frames, 2), dtype=np.float32)

        # While priming, stay silent and let the writer build the cushion
        # up. Without this the very first pull empties the buffer and the
        # reader spends the rest of the session at zero available, where
        # any callback jitter is a dropout instead of a glitch.
        if self._monitor_priming:
            if self._monitor_available < self._monitor_prefill:
                return out
            self._monitor_priming = False

        n = min(frames, self._monitor_available)
        if n > 0:
            end = self._monitor_read + n
            if end <= cap:
                out[:n] = self._monitor_buf[self._monitor_read:end]
            else:
                first = cap - self._monitor_read
                out[:first] = self._monitor_buf[self._monitor_read:]
                out[first:n] = self._monitor_buf[:n - first]
            self._monitor_read = end % cap
            self._monitor_available -= n
        if n < frames:
            # Zero-fill the shortfall and carry on - deliberately *not*
            # re-priming, which would mute a further PREFILL seconds for
            # what may have been one late callback. This is self-healing
            # anyway: the pull covered `frames` worth of wall-clock time
            # while consuming only `n` samples, so the writer gains the
            # difference back and the cushion rebuilds on its own.
            self.monitor_underruns += 1
        return out * db_to_linear(self.monitor_gain_db)

    def adjust_gain(self, delta_db):
        """Adjusts whichever gain is currently the relevant one to tune:
        the live monitor if it's on (most likely what you're listening to
        and want to adjust), otherwise whichever of amp/Mac the loop is
        routed to."""
        with self.lock:
            if self.monitor_enabled:
                self.monitor_gain_db += delta_db
                save_gain_db(MONITOR_GAIN_FILE, self.monitor_gain_db)
                return "monitor", self.monitor_gain_db
            elif self.output_target == OUTPUT_TO_AMP:
                self.gain_db += delta_db
                save_gain_db(GAIN_FILE, self.gain_db)
                return self.output_target, self.gain_db
            else:
                self.mac_gain_db += delta_db
                save_gain_db(MAC_GAIN_FILE, self.mac_gain_db)
                return self.output_target, self.mac_gain_db

    def toggle_output_target(self):
        with self.lock:
            self.output_target = (
                OUTPUT_TO_MAC if self.output_target == OUTPUT_TO_AMP else OUTPUT_TO_AMP
            )
            return self.output_target

    def _advance_playhead(self, frames, gain):
        """Pulls the next `frames` samples of loop playback (gain-applied),
        wrapping around, and moves the shared playhead forward. Whichever
        output stream is actually the active target calls this - the
        other stays silent and leaves the playhead untouched, so switching
        targets resumes from wherever playback left off rather than
        restarting or jumping.

        Indexed modulo the loop length rather than sliced, so a loop
        *shorter than one callback block* tiles correctly instead of
        returning a short array. That case is not hypothetical: the
        streams are opened with blocksize=0, so PortAudio picks the block
        size (measured ~4096 frames / 93ms on the Katana, 512 on the Mac
        output) and may vary it, while a quick record->stop - trivially
        easy with mouth-triggering at a 0.05s debounce - can capture less
        than that. The old slice-and-concatenate returned fewer than
        `frames` samples there, and the resulting broadcast ValueError,
        raised inside the callback, makes PortAudio abort the stream for
        good."""
        buf = self.loop_buffer
        n = len(buf)
        if n == 0:
            return np.zeros((frames, 2), dtype=np.float32)
        idx = (self.playhead + np.arange(frames)) % n
        self.playhead = (self.playhead + frames) % n
        return buf[idx] * gain

    def callback(self, indata, outdata, frames, time_info, status):
        """Main duplex stream, tied to the Katana - always handles
        recording (the guitar signal only comes in through here) and, when
        output_target is 'amp', loop playback out to the amp's speaker."""
        # Deliberately no printing from here: once xruns start, a print per
        # callback from the realtime thread stalls it further and turns
        # occasional glitching into sustained dropout. Counters instead;
        # main.py surfaces them via monitor_status().
        self.cb_count += 1
        if status:
            self.cb_status_count += 1
            self.last_status = str(status)

        outdata.fill(0)
        # An exception escaping a PortAudio callback aborts the stream
        # permanently - the rest of the session is then silent with no
        # error anywhere, which is exactly the failure mode this file has
        # already been debugged for once. Nothing in here is worth losing
        # audio over: record the error, output silence for this block, and
        # keep the stream alive so it's recoverable and visible.
        try:
            if self.input_listener is not None:
                self.input_listener(indata[:, INPUT_SLICE])
            if self.dry_listener is not None:
                self.dry_listener(indata[:, DRY_SLICE])
            with self.lock:
                if self.monitor_enabled:
                    self._monitor_push(indata[:, INPUT_SLICE])
                if self.state == RECORDING:
                    if not self.record_chunks:
                        self.record_start_time = (time.perf_counter() - time_info.currentTime
                                                  + time_info.inputBufferAdcTime)
                    block = indata[:, INPUT_SLICE].copy()
                    if self._stop_samples is not None:
                        have = sum(len(c) for c in self.record_chunks)
                        if have + len(block) >= self._stop_samples:
                            self.record_chunks.append(block[:self._stop_samples - have])
                            self._finish_locked()
                        else:
                            self.record_chunks.append(block)
                    else:
                        self.record_chunks.append(block)
                elif (self.state == PLAYING and self.loop_buffer is not None
                        and self.output_target == OUTPUT_TO_AMP):
                    if self.play_start_time is None and self.playhead == 0:
                        self.play_start_time = (time.perf_counter() - time_info.currentTime
                                                + time_info.outputBufferDacTime
                                                + OUTPUT_EXTRA_DELAY)
                    gain = db_to_linear(self.gain_db)
                    outdata[:, OUTPUT_SLICE] = self._advance_playhead(frames, gain)
            if self.output_source is not None:
                extra = self.output_source(frames, time_info)
                if extra is not None:
                    outdata[:, OUTPUT_SLICE] += extra[:, None] * db_to_linear(self.gain_db)
        except Exception as exc:
            outdata.fill(0)
            self.cb_errors += 1
            self.last_error = f"katana: {type(exc).__name__}: {exc}"

    def mac_callback(self, outdata, frames, time_info, status):
        """Second, output-only stream on the Mac's default output device.
        Mixes two independent things, either/both of which may be active:
        loop playback (only when output_target is 'mac') and the live
        guitar monitor (only when monitor_enabled) - otherwise silent, so
        it's safe to leave running all the time rather than opening/
        closing a stream on every toggle (which would click)."""
        self.mac_cb_count += 1
        if status:
            self.mac_cb_status_count += 1
            self.last_status = str(status)

        outdata.fill(0)
        # See callback() - an exception here would abort this stream and
        # take the live monitor down with it for the rest of the session,
        # while the Katana stream carried on working normally.
        try:
            with self.lock:
                if (self.state == PLAYING and self.loop_buffer is not None
                        and self.output_target == OUTPUT_TO_MAC):
                    gain = db_to_linear(self.mac_gain_db)
                    outdata[:, :2] += self._advance_playhead(frames, gain)
                if self.monitor_enabled:
                    outdata[:, :2] += self._monitor_pull(frames)
        except Exception as exc:
            outdata.fill(0)
            self.mac_cb_errors += 1
            self.last_error = f"mac: {type(exc).__name__}: {exc}"

    def toggle_record(self):
        with self.lock:
            if self.state != RECORDING:
                self.record_chunks = []
                self.state = RECORDING
                print("Recording... press 'r' again to stop and loop.")
            else:
                if self.record_chunks:
                    self.loop_buffer = np.concatenate(self.record_chunks, axis=0)
                    self.playhead = 0
                    self.state = PLAYING
                    secs = len(self.loop_buffer) / SAMPLE_RATE
                    print(f"Looping {secs:.2f}s. Press 'r' to record a new "
                          f"loop, 's' to stop/clear.")
                else:
                    self.state = IDLE

    def stop(self):
        with self.lock:
            self.state = IDLE
            self.loop_buffer = None
            self.playhead = 0
            print("Stopped, loop cleared.")

    def _finish_locked(self, length=None):
        """Recording -> playing, keeping `length` samples if given. Caller
        holds self.lock."""
        self._stop_samples = None
        if not self.record_chunks:
            self.state = IDLE
            return
        buffer = np.concatenate(self.record_chunks, axis=0)
        self.loop_buffer = buffer[:length] if length else buffer
        self.playhead = 0
        self.play_start_time = None
        self.state = PLAYING
        print(f"Looping {len(self.loop_buffer) / SAMPLE_RATE:.2f}s.")

    def recorded_seconds(self):
        with self.lock:
            return sum(len(c) for c in self.record_chunks) / SAMPLE_RATE

    def finish_recording(self, seconds):
        """Stop recording with the loop exactly `seconds` long: trimmed now
        if that much is already recorded, else recording carries on until
        it is (so a loop can be rounded to whole bars either way)."""
        target = max(1, int(round(seconds * SAMPLE_RATE)))
        with self.lock:
            if self.state != RECORDING:
                return
            if sum(len(c) for c in self.record_chunks) >= target:
                self._finish_locked(target)
            else:
                self._stop_samples = target

    def cycle(self):
        """Single-trigger pedal-style control: idle -> recording -> playing
        -> idle (cleared) -> recording -> ... One gesture/keypress per step,
        no separate start/stop/clear actions to remember."""
        with self.lock:
            if self.state == IDLE:
                self.record_chunks = []
                self.record_start_time = None
                self._stop_samples = None
                self.state = RECORDING
                print("Recording...")
            elif self.state == RECORDING:
                self._finish_locked()
            elif self.state == PLAYING:
                self.state = IDLE
                self.loop_buffer = None
                self.playhead = 0
                self.play_start_time = None
                print("Stopped, loop cleared.")

    def open_stream(self, latency=None):
        """latency=None keeps PortAudio's default ('high': ~93ms blocks,
        ~96ms in / ~188ms out), which the looper and monitor here were
        tuned with. 'low' measured 7ms blocks, ~13ms in / ~16ms out, with
        no xruns under load - what the drum/bass tools use, since the bass
        has to be scheduled further ahead the slower the output is."""
        return sd.Stream(
            device=(DEVICE_NAME, DEVICE_NAME),
            samplerate=SAMPLE_RATE,
            channels=(IN_CHANNELS, OUT_CHANNELS),
            dtype="float32",
            callback=self.callback,
            latency=latency,
        )

    def open_mac_stream(self):
        """Output-only stream on the system's current default output
        device (device=None picks it up automatically, same as whatever
        macOS's own default output is).

        No latency='high' here: it was tried against the silence bug and
        did nothing, because the problem was never PortAudio's own
        buffering - it was the monitor ring buffer running permanently
        empty (see MONITOR_PREFILL_SECONDS, which is where the jitter
        margin actually comes from now). Stacking a high-latency device
        buffer on top of the prefill would only add delay to what's
        supposed to be a live monitor."""
        return sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            dtype="float32",
            callback=self.mac_callback,
        )
