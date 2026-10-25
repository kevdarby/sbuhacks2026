"""Plays a groove on the relay locked to the beat a TempoTracker hears in
the guitar (see amp/tempo.py). Starts once the tracker has held a tempo
for a moment, follows it as it drifts, and stops when the playing stops.
"""

import math
import threading
import time

from amp.bass import MIN_BASS_AHEAD, SCHEDULE_AHEAD
from drums.patterns import PULSES, STYLES
from drums.relay import RELAY_RECOVERY

# The tracker has to hold a tempo this long before the drums come in, so a
# few stray notes don't start them off at a guessed tempo
LOCK_IN_SECONDS = 1.0
# How the drums' beat grid is pulled toward the tracker's, per beat: shift
# by this fraction of the timing error, and adjust the tempo by this
# fraction of it too. Correcting the tempo from the timing error (rather
# than copying the tracker's BPM) is what lets a constant lag shrink to
# zero - copying the BPM left the drums a steady ~20ms behind in testing,
# from the BPM estimate being a fraction of a percent off. Low enough that
# one off estimate doesn't make the drums lurch, high enough to follow a
# player who speeds up.
PHASE_CORRECTION = 0.3
TEMPO_CORRECTION = 0.1
# A tempo change bigger than this is jumped to at once rather than slid
# over to, which would take several beats of audibly wrong timing
JUMP_RATIO = 0.08
# Fire this much early: the time from sending the relay command to the
# drum actually sounding - serial write, relay close and the plunger's
# travel. Measured with the MacBook mic at 55-58ms, the same for every
# pulse length from 10ms to 80ms.
DEFAULT_LEAD = 0.057
SILENT_PULSE = 0.015  # shorter than this barely moves the plunger
# time.sleep() was measured waking ~5ms late here (the tracker and audio
# threads compete for the GIL), so sleep to just short of the target and
# spin the rest
SPIN_SECONDS = 0.006


def _sleep_until(target):
    delay = target - time.perf_counter() - SPIN_SECONDS
    if delay > 0:
        time.sleep(delay)
    while time.perf_counter() < target:
        pass


class BeatFollower:
    def __init__(self, tracker, relay, style="rock", fill=True, lead=DEFAULT_LEAD, arranger=None,
                 bass=None):
        """With an `arranger` (drums/arranger.py) the groove is rewritten
        every bar to fit the guitar, and `style` is only what plays until
        there's a bar of guitar to read; the arranger does its own fills.
        With a `bass` (amp/bass.py BassLine) a bass line is played along
        with the groove."""
        self.tracker = tracker
        self.relay = relay
        self.groove, self.fill_pattern = STYLES[style]
        self.arranger = arranger
        self.bass = bass
        self.fill = fill and arranger is None
        self.lead = lead
        self.playing = False
        self.bpm = None
        self._thread = None
        self._bass_thread = None
        self._running = False
        # The beat grid as of the latest step, for the bass thread:
        # (time of step `step`, step, seconds per step), or None when stopped
        self._grid = None
        # Paused: keep following the tempo but play nothing - while a loop
        # replays the band's own recorded part (drums/loopband.py)
        self.paused = False
        # Muted by the player (the 'p' key): the same, but separate from
        # `paused` so the loop replay can't unmute it - and it silences the
        # replay too
        self.muted = False
        # Called with (time the hit sounds, pulse) for every drum hit played
        self.on_hit = None

    def _pattern(self, step):
        bar = step // 16
        return self.fill_pattern if self.fill and bar % 4 == 3 else self.groove

    def _pulse(self, step, step_seconds):
        """Pulse for this step, shortened if the next hit comes too soon for
        the relay to reopen in time - so any groove survives a fast tempo,
        just with its accents softened."""
        pulse = PULSES.get(self._pattern(step)[step % 16])
        if pulse is None:
            return None
        gap = 1
        while gap < 16 and self._pattern(step + gap)[(step + gap) % 16] not in PULSES:
            gap += 1
        pulse = min(pulse, gap * step_seconds - RELAY_RECOVERY)
        return pulse if pulse >= SILENT_PULSE else None

    def _plan_bass_step(self, step, step_time, step_seconds):
        """Schedule the bass note (if any) for `step`, which sounds at
        `step_time`."""
        if step % 4 == 0:
            self.bass.beat(step // 4, step_time, 4 * step_seconds)

        def plays(s):
            return self.bass.plays_on(self._pattern(s)[s % 16], s % 16)

        if not plays(step):
            return
        # Hold right up to the next bass note (up to half a bar); the
        # synth's release overlaps it, so the line is legato
        gap = 1
        while gap < 8 and not plays(step + gap):
            gap += 1
        self.bass.plan(step_time, gap * step_seconds, 4 * step_seconds, step // 4, step % 16)

    def _run_bass(self):
        """Decides each bass note as late as the output latency allows -
        SCHEDULE_AHEAD before it sounds - so it's chosen from the freshest
        guitar audio. Planned from the drum loop instead (as at first), notes
        were decided right after the previous drum hit and then a full step
        further ahead: ~0.3s early at 120 BPM, which left the bass on the
        previous chord for 21 of 48 notes in a recorded take of a song with
        quick changes."""
        last = None
        while self._running:
            grid = self._grid
            if grid is None:
                last = None
                time.sleep(0.005)
                continue
            anchor_time, anchor_step, step_seconds = grid
            now = time.perf_counter()
            # The furthest step that's now within SCHEDULE_AHEAD
            target = anchor_step + math.floor((now + SCHEDULE_AHEAD - anchor_time) / step_seconds)
            # After a jump in the step count (a tempo jump or beat-1 shift)
            # don't try to catch up on steps that are already too late
            first = target if last is None or target - last > 4 else last + 1
            for s in range(first, target + 1):
                step_time = anchor_time + (s - anchor_step) * step_seconds
                if step_time > now + MIN_BASS_AHEAD and not (self.paused or self.muted):
                    self._plan_bass_step(s, step_time, step_seconds)
            last = max(last if last is not None else target, target)
            time.sleep(0.004)

    def _run(self):
        locked_since = None
        last_estimate = None
        step = 0
        period = None
        next_time = None
        while self._running:
            estimate = self.tracker.estimate
            now = time.perf_counter()

            if estimate is None:
                if self.playing and self.bass is not None:
                    self.bass.synth.clear()  # don't let the queued notes ring on
                self._grid = None
                locked_since = None
                self.playing = False
                self.bpm = None
                time.sleep(0.02)
                continue

            bpm, beat_time, _ = estimate
            est_period = 60 / bpm

            if not self.playing:
                locked_since = locked_since or now
                if now - locked_since < LOCK_IN_SECONDS:
                    time.sleep(0.02)
                    continue
                # Come in on the next beat the tracker predicts
                period = est_period
                beats_ahead = int((now + 0.1 - beat_time) / period) + 1
                next_time = beat_time + beats_ahead * period
                step = 0
                last_estimate = estimate
                if self.arranger is not None:
                    self.arranger.reset()
                if self.bass is not None:
                    self.bass.reset()
                self.playing = True
            elif estimate is not last_estimate and step % 4 == 0:
                last_estimate = estimate
                # Compare our next beat with the nearest beat the tracker
                # predicts
                predicted = beat_time + round((next_time - beat_time) / est_period) * est_period
                error = predicted - next_time
                if abs(est_period / period - 1) > JUMP_RATIO:
                    # The player has moved to a clearly different tempo:
                    # jump straight onto the tracker's grid and keep
                    # playing. Stopping and re-locking here (as at first)
                    # left 1.5-2s holes every time someone playing freely
                    # drifted 8%, and threw away where beat 1 was.
                    period = est_period
                    next_time = predicted
                    # Never schedule into the past; skip whole beats so the
                    # position in the bar stays right
                    while next_time < time.perf_counter() + 0.01:
                        next_time += period
                        step += 4
                else:
                    # Small drift: nudge our grid toward the tracker's
                    next_time += PHASE_CORRECTION * error
                    period += TEMPO_CORRECTION * error

            self.bpm = 60 / period
            step_seconds = period / 4
            if self.arranger is not None and step % 16 == 0:
                self.groove, shift = self.arranger.next_bar(next_time, step_seconds)
                if shift:
                    # The player's beat 1 is `shift` beats into our bar, so
                    # we're actually partway through theirs
                    step += (16 - 4 * shift) % 16
                    if self.bass is not None:
                        self.bass.reset()  # its beat numbering just moved
            self._grid = (next_time, step, step_seconds)
            _sleep_until(next_time - self.lead)
            # If this thread got held up past half a step, skip the hit
            # rather than play it audibly late
            if not (self.paused or self.muted) and time.perf_counter() - (next_time - self.lead) < step_seconds / 2:
                pulse = self._pulse(step, step_seconds)
                if pulse is not None:
                    if self.on_hit is not None:
                        self.on_hit(next_time, pulse)
                    self.relay.hit(pulse)
            step += 1
            next_time += step_seconds

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self.bass is not None:
            self._bass_thread = threading.Thread(target=self._run_bass, daemon=True)
            self._bass_thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        if self._bass_thread is not None:
            self._bass_thread.join()
