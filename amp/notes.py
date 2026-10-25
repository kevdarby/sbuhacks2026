"""Which notes are sounding, from Spotify's Basic Pitch neural network
(models/basic_pitch, Apache-2.0 - see the LICENSE there). Used by
amp/chords.py to find chords.

Basic Pitch was trained on real instruments to report the notes actually
played rather than their overtones - the exact thing the plain pitch-class
profile gets wrong on guitar (the B strings in open Em/G drowning out the
root). On the Katana's dry channels (ch3/4) it read a recorded Em-G-D-A
take essentially perfectly where the template detector kept hearing Em as
G or Bm. It's no better on the distorted channels 1/2, so it's fed dry
audio.

Only the model file is vendored, run with onnxruntime: the basic-pitch
package itself pulls in TensorFlow. One ~2s window takes ~10ms.
"""

import threading
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "basic_pitch" / "nmp.onnx"
MODEL_RATE = 22050
WINDOW_SAMPLES = 43844        # the model's fixed input: 2s minus one hop
HOP = 256                     # model frames are this many samples apart
LOWEST_MIDI = 21              # output column 0 is A0
# Basic Pitch pads this much silence before audio (half its window overlap),
# so audio placed after it sits where the model expects it, away from the
# window's unreliable edge
LEAD_IN = 3840
NOTE_OUTPUT = "StatefulPartitionedCall:1"   # sustained-note activations


def available():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return MODEL_PATH.exists()


class NoteModel:
    def __init__(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        # It runs beside the audio and video threads; one or two cores is
        # plenty for a 10ms job and leaves the rest alone
        options.intra_op_num_threads = 2
        self._session = ort.InferenceSession(str(MODEL_PATH), options,
                                             providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        self._lock = threading.Lock()

    def pitch_classes(self, audio, sample_rate):
        """12-bin profile (C through B) of how strongly each pitch class is
        being played in `audio` (up to ~1.8s), normalized to sum 1, or None
        if no notes are detected."""
        if sample_rate == 2 * MODEL_RATE:
            # Average pairs before dropping every other sample: a crude
            # low-pass, enough to keep aliasing out of the guitar's range
            audio = audio[:len(audio) // 2 * 2].reshape(-1, 2).mean(axis=1)
        else:
            idx = np.arange(0, len(audio), sample_rate / MODEL_RATE).astype(int)
            audio = audio[idx]
        audio = audio[-(WINDOW_SAMPLES - LEAD_IN):]
        peak = np.abs(audio).max()
        if peak <= 0:
            return None
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        # The dry signal is ~10x quieter than a normal recording; scale it to
        # a typical level for the model
        window[LEAD_IN:LEAD_IN + len(audio)] = audio / peak * 0.5
        with self._lock:
            notes = self._session.run([NOTE_OUTPUT], {self._input: window[None, :, None]})[0][0]
        first, last = LEAD_IN // HOP, (LEAD_IN + len(audio)) // HOP
        activation = notes[first:last].sum(axis=0)
        profile = np.zeros(12)
        np.add.at(profile, (np.arange(len(activation)) + LOWEST_MIDI) % 12, activation)
        total = profile.sum()
        return profile / total if total > 0 else None
