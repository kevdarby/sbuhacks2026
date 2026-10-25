"""The band and the loop pedal, together. While a loop records, the drums
and bass follow the guitar live as usual, and every drum hit is logged.
When recording stops the loop is rounded to whole bars at the band's
tempo, and while it plays the band stops following and replays its own
recorded part on every pass, in time with the loop audio.

Only the drums are replayed: the Katana mixes its USB playback into the
channels the looper records (see DRY_ANALYSIS_GAIN in amp/audio.py), so
the bass is already in the loop recording, in sync. The solenoid isn't.
"""

import threading
import time

from amp.audio import PLAYING, RECORDING, SAMPLE_RATE
from drums.follower import SILENT_PULSE, _sleep_until
from drums.relay import RELAY_RECOVERY

# A loop shorter than this fraction of a bar isn't rounded to bars - it's
# probably a mistake or a deliberate sound effect, not a riff
MIN_BARS_TO_ROUND = 0.5


class LoopBand:
    def __init__(self, looper, follower, relay):
        self.looper = looper
        self.follower = follower
        self.relay = relay
        follower.on_hit = self._on_hit
        self._recording = False
        self._hits = []           # (time the hit sounded, pulse)
        self._last_state = looper.state
        self._replaying = False
        self._thread = None
        # The band's tempo when the loop was cut to bars, or None if it wasn't
        # - the live tempo can disappear while the loop plays (stop strumming
        # and the tracker goes quiet)
        self.loop_bpm = None

    def _on_hit(self, sound_time, pulse):
        if self._recording:
            self._hits.append((sound_time, pulse))

    def bar_seconds(self):
        grid = self.follower._grid
        if not self.follower.playing or grid is None:
            return None
        return 16 * grid[2]

    def trigger(self):
        """The loop pedal: what the mouth gesture / spacebar call. Same
        record -> loop -> stop cycle as Looper.cycle(), except that stopping
        a recording rounds the loop to whole bars when the band has a tempo -
        so the drums' groove carries on unbroken where the loop wraps."""
        if self.looper.state != RECORDING:
            self.looper.cycle()
            return
        bar = self.bar_seconds()
        recorded = self.looper.recorded_seconds()
        if bar is None or recorded < MIN_BARS_TO_ROUND * bar:
            self.loop_bpm = None
            self.looper.cycle()
            return
        self.loop_bpm = 60 / (bar / 4)
        bars = max(1, round(recorded / bar))
        self.looper.finish_recording(bars * bar)
        print(f"Loop: {bars} bar{'s' * (bars > 1)} ({bars * bar:.2f}s at {60 / (bar / 4):.0f} BPM)")

    def update(self):
        """Call regularly (every video frame): follows the looper's state."""
        state = self.looper.state
        if state == self._last_state:
            return
        if state == RECORDING:
            self._hits = []
            self._recording = True
        elif state == PLAYING:
            self._recording = False
            self._start_replay()
        else:  # IDLE: loop cleared, back to following live
            self._recording = False
            self._stop_replay()
        self._last_state = state

    def _start_replay(self):
        self._stop_replay()
        # Stop playing live straight away; the replay thread waits for the
        # loop's first block to know when it's heard
        self.follower.paused = True
        self._replaying = True
        self._thread = threading.Thread(target=self._replay, daemon=True)
        self._thread.start()

    def _stop_replay(self):
        self._replaying = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.follower.paused = False

    def _replay(self):
        while self._replaying and self.looper.play_start_time is None:
            if self.looper.state != PLAYING:
                return
            time.sleep(0.002)
        if not self._replaying or self.looper.loop_buffer is None:
            return
        length = len(self.looper.loop_buffer) / SAMPLE_RATE
        heard_start = self.looper.play_start_time
        record_start = self.looper.record_start_time
        # Each hit's place in the loop: the same distance from the loop's
        # first sample as it was from the first recorded sample
        events = sorted((t - record_start, pulse) for t, pulse in self._hits
                        if 0 <= t - record_start < length)
        if not events:
            return
        lead = self.follower.lead
        loop = 0
        while self._replaying:
            for i, (offset, pulse) in enumerate(events):
                sound_time = heard_start + loop * length + offset
                fire_at = sound_time - lead
                if fire_at < time.perf_counter() - 0.01:
                    continue  # missed (just started, or the thread was held up)
                while self._replaying and fire_at - time.perf_counter() > 0.05:
                    time.sleep(0.02)
                if not self._replaying:
                    return
                _sleep_until(fire_at)
                # Keep the relay able to reopen before the next hit, across
                # the loop's wrap too
                next_offset = events[i + 1][0] if i + 1 < len(events) else events[0][0] + length
                pulse = min(pulse, next_offset - offset - RELAY_RECOVERY)
                if pulse >= SILENT_PULSE and not self.follower.muted:
                    self.relay.hit(pulse)
            loop += 1

    def stop(self):
        self._stop_replay()
