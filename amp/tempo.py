"""Live tempo and beat tracking from the guitar signal, via the same USB
audio input the looper and tuner read (see amp/audio.py). Like the tuner,
it's pure signal analysis and never talks to the amp.

Three stages, each a standard technique:

1. Onset strength - spectral flux: how much louder each frequency bin got
   since the previous hop, summed over bins. Picks out picked notes and
   strums, including a new chord at the same overall loudness, which a plain
   volume envelope would miss.
2. Tempo - autocorrelation of the onset-strength envelope over the last
   several seconds. Playing at a steady tempo makes the envelope correlate
   with itself at one beat's lag. Every multiple of the beat correlates too
   (half/double tempo), so lags are weighted toward a typical tempo and
   credited for a matching peak at twice the lag.
3. Fine tempo and beat phase - around that coarse tempo, slide a comb of
   beats over the recent envelope for each nearby period and offset, and
   keep the one that lands on the most onset energy. That pins the tempo
   much more precisely than the autocorrelation can, and gives where the
   beats actually fall, not just how far apart they are.
"""

import collections
import threading
import time

import numpy as np

HOP = 512           # ~11.6ms per envelope sample at 44.1kHz
FRAME = 1024
WINDOW_SECONDS = 8.0
MIN_SECONDS = 3.0   # don't guess a tempo from less than this
MIN_BPM = 60.0
MAX_BPM = 160.0
# Log-gaussian prior over tempo, centered on a typical strumming tempo. One
# octave wide - enough to prefer 100 over 50 or 200 without overruling a
# clearly played 70 or 150.
PRIOR_CENTER_BPM = 110.0
PRIOR_OCTAVES = 1.0
# Peak autocorrelation / zero-lag below this means no steady pulse
MIN_CONFIDENCE = 0.15
# Period fine-tuning (see _refine): search +-4% around the coarse estimate
# in 0.25% steps, weighting each older beat 0.9x the next. Picked by
# replaying recorded takes (tools/record_take.py) against their click.
REFINE_SPAN = 0.04
REFINE_STEP = 0.0025
COMB_DECAY = 0.9
SILENCE_RMS = 0.003
# How long without playing before the beat is considered lost
SILENCE_TIMEOUT = 2.0
# How long before the audio callback runs the last sample of its block was
# captured - measured from the Katana stream's time_info (callback time
# minus ADC time, less one 4096-frame block)
CAPTURE_DELAY = 0.006


class TempoTracker:
    """Fed raw audio from the realtime callback (feed() only copies it); the
    analysis runs in update(), called from its own thread via start(), so
    the FFTs never run on the audio thread.

    Times are reported on the time.perf_counter() clock. Audio arrives in
    ~93ms bursts (see MONITOR_PREFILL_SECONDS in amp/audio.py), so each
    sample's capture time is reconstructed from the sample count plus the
    earliest arrival offset seen recently: a block can show up late, never
    early, so the minimum is the least-jittered estimate."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.hop_seconds = HOP / sample_rate
        self.window_hops = int(WINDOW_SECONDS / self.hop_seconds)
        self.min_hops = int(MIN_SECONDS / self.hop_seconds)
        self.lock = threading.Lock()

        self._pending = []
        self._samples_fed = 0
        self._arrival_offsets = collections.deque(maxlen=50)
        self._clock_offset = None

        self._carry = np.zeros(0, dtype=np.float32)
        self._prev_mag = None
        self._window = np.hanning(FRAME).astype(np.float32)
        self._envelope = collections.deque(maxlen=self.window_hops)
        self._hops_done = 0
        self._last_active_hop = None

        self._bpm = None
        self._disagreements = 0
        self.estimate = None  # (bpm, beat_time, confidence) or None
        self._thread = None
        self._running = False

    def feed(self, stereo_block, now=None):
        now = time.perf_counter() if now is None else now
        mono = stereo_block.mean(axis=1).astype(np.float32)
        with self.lock:
            self._pending.append(mono)
            self._samples_fed += len(mono)
            captured = now - CAPTURE_DELAY
            self._arrival_offsets.append(captured - self._samples_fed / self.sample_rate)
            self._clock_offset = min(self._arrival_offsets)

    def sample_time(self, sample_index):
        """perf_counter time at which a given input sample was captured."""
        return self._clock_offset + sample_index / self.sample_rate

    def time_to_sample(self, t):
        """Input sample index captured at perf_counter time t (fractional),
        or None before any audio has arrived. Lets other analyzers fed the
        same blocks (amp/chords.py) share this clock."""
        with self.lock:
            if self._clock_offset is None:
                return None
            return (t - self._clock_offset) * self.sample_rate

    def _consume(self):
        with self.lock:
            blocks, self._pending = self._pending, []
        if not blocks:
            return
        audio = np.concatenate([self._carry, *blocks])
        # Frames are FRAME long but step by HOP, so keep the tail that the
        # next frame still overlaps
        n_hops = max(0, (len(audio) - FRAME) // HOP + 1)
        fluxes = []
        for i in range(n_hops):
            frame = audio[i * HOP:i * HOP + FRAME]
            mag = np.log1p(100 * np.abs(np.fft.rfft(frame * self._window)))
            if self._prev_mag is None:
                flux = 0.0
            else:
                flux = float(np.maximum(mag - self._prev_mag, 0).sum())
            self._prev_mag = mag
            fluxes.append(flux)
            if np.sqrt(np.mean(frame ** 2)) >= SILENCE_RMS:
                self._last_active_hop = self._hops_done + i
        # Under the lock because onset_envelope() reads these from the
        # drum thread
        with self.lock:
            self._envelope.extend(fluxes)
            self._hops_done += n_hops
        self._carry = audio[n_hops * HOP:]

    def onset_envelope(self):
        """The recent onset-strength envelope and the perf_counter time of
        each sample, for reading the rhythm being played (see
        drums/arranger.py). Empty until audio has arrived."""
        with self.lock:
            env = np.asarray(self._envelope, dtype=np.float64)
            first_hop = self._hops_done - len(env)
            if self._clock_offset is None:
                return env, np.zeros(0)
            # Same mid-frame onset position update() uses for beat times
            samples = (first_hop + np.arange(len(env))) * HOP + FRAME // 2
            return env, self.sample_time(samples)

    def _estimate_period(self, env):
        """Beat period in hops (fractional) and its confidence."""
        x = env - env.mean()
        n = len(x)
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
        if ac[0] <= 0:
            return None, 0.0
        # Unbiased: a longer lag has fewer overlapping samples, so its raw
        # sum is smaller even for a perfectly steady pulse (the same trap
        # amp/tuner.py describes for pitch)
        ac = np.maximum(ac / (n - np.arange(n)), 0)
        ac = ac / ac[0]

        def ac_at(lag):
            # Linear interpolation, since half a lag lands between samples
            lag = np.clip(lag, 0, n - 2)
            i = np.floor(lag).astype(int)
            frac = lag - i
            return ac[i] * (1 - frac) + ac[i + 1] * frac

        min_lag = int(60 / MAX_BPM / self.hop_seconds)
        max_lag = min(int(60 / MIN_BPM / self.hop_seconds) + 1, n // 2 - 1)
        lags = np.arange(min_lag, max_lag + 1).astype(float)
        bpms = 60 / (lags * self.hop_seconds)
        prior = np.exp(-0.5 * (np.log2(bpms / PRIOR_CENTER_BPM) / PRIOR_OCTAVES) ** 2)
        # A real beat lag is backed up by a peak at twice the lag (the
        # pulse continues) and at half of it (eighth notes in between).
        # Checking only twice the lag let slow eighth-note strumming lock
        # at double time, and fast strumming at 2/3 of the tempo; checking
        # half the lag too fixed both in testing on synthetic strums.
        score = prior * (ac_at(lags) + 0.5 * ac_at(2 * lags) + 0.5 * ac_at(lags / 2))
        best = int(np.argmax(score))
        lag = float(lags[best])
        # Parabolic interpolation around the peak for a sub-hop period
        if 0 < best < len(score) - 1:
            a, b, c = score[best - 1], score[best], score[best + 1]
            denom = a - 2 * b + c
            if denom < 0:
                lag += 0.5 * (a - c) / denom
        return lag, float(ac_at(lag))

    def _refine(self, env, coarse):
        """Fine-tune the period around the autocorrelation's coarse guess
        and find where the beats fall, together: slide a comb of beats
        (plus half-weight teeth on the off-beats) over the envelope for
        each nearby period and phase, and keep the one landing on the most
        onset energy. The autocorrelation alone only pins the tempo to
        about 2% (one hop is 2% of a beat at 100 BPM), and that much error
        moves the drums ~12ms per beat. Returns (period, hop index within
        env of the most recent beat)."""
        n = len(env)
        # Smooth so a beat landing between envelope samples still scores
        kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
        smooth = np.convolve(env, kernel / kernel.sum(), mode="same")
        xs = np.arange(n)
        best_score, best_period, best_phase = -1.0, coarse, 0.0
        for period in coarse * (1 + np.arange(-REFINE_SPAN, REFINE_SPAN + 1e-9, REFINE_STEP)):
            beats = np.arange(int((n - 2) / period))
            phases = np.arange(0, period, 0.5)
            # Recent beats weighted more, so the phase reflects where the
            # player is now rather than where they were 8 seconds ago
            weights = COMB_DECAY ** beats
            on = (n - 1 - phases[:, None]) - beats[None, :] * period
            off = on - period / 2
            score = ((np.interp(on, xs, smooth) * (on >= 0)) * weights).sum(1)
            score += 0.5 * ((np.interp(off, xs, smooth) * (off >= 0)) * weights).sum(1)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best_score, best_period, best_phase = score[i], period, phases[i]
        return best_period, n - 1 - best_phase

    def update(self):
        self._consume()
        if (self._last_active_hop is None
                or (self._hops_done - self._last_active_hop) * self.hop_seconds > SILENCE_TIMEOUT):
            self._bpm = None
            with self.lock:
                self._envelope.clear()
            self.estimate = None
            return
        if len(self._envelope) < self.min_hops:
            return

        with self.lock:
            env = np.asarray(self._envelope, dtype=np.float64)
        period, confidence = self._estimate_period(env)
        if period is None:
            return
        # Confidence only gates the first lock. Real strumming scored
        # 0.13-0.22 in recorded takes, so gating every update (as at first)
        # threw most of them away and left the drums running uncorrected.
        if self._bpm is None and confidence < MIN_CONFIDENCE:
            return
        bpm = 60 / (period * self.hop_seconds)

        if self._bpm is not None:
            # A half/double-time reading of the same pulse is folded back to
            # the locked tempo rather than treated as a new one - in the 120
            # BPM take the raw estimate kept dropping to 60. Not 2:3 as well:
            # that kept a wrong 2:3 first lock wrong forever.
            for ratio in (2.0, 0.5):
                if abs(bpm * ratio / self._bpm - 1) < 0.06:
                    bpm *= ratio
                    break
        # Hold onto a locked tempo: a genuinely different one only wins if
        # it keeps winning
        if self._bpm is None or abs(bpm / self._bpm - 1) < 0.08:
            self._bpm = bpm
            self._disagreements = 0
        else:
            self._disagreements += 1
            if self._disagreements >= 6:
                self._bpm = bpm
                self._disagreements = 0

        period, beat_hop = self._refine(env, 60 / self._bpm / self.hop_seconds)
        # Each refinement can move the tempo up to REFINE_SPAN, so over many
        # updates it could creep outside the allowed range - a recorded take
        # walked from 157 up to 184, the eighth-note pulse of a ~92 BPM song.
        # Fold back by octaves; beat_hop is still a beat either way.
        while 60 / (period * self.hop_seconds) > MAX_BPM:
            period *= 2
        while 60 / (period * self.hop_seconds) < MIN_BPM:
            period /= 2
        self._bpm = 60 / (period * self.hop_seconds)
        first_hop = self._hops_done - len(env)
        # Envelope sample k describes the frame starting at k*HOP; its
        # onset sits about mid-frame
        beat_sample = (first_hop + beat_hop) * HOP + FRAME // 2
        with self.lock:
            beat_time = self.sample_time(beat_sample)
        self.estimate = (self._bpm, beat_time, confidence)

    def start(self, interval=0.25):
        self._running = True

        def run():
            while self._running:
                self.update()
                time.sleep(interval)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
