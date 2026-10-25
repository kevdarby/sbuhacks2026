import json


class Calibration:
    """Manual two-point calibration: record a ratio for the "low" state
    (closed fist / closed mouth) and the "high" state (open hand / open
    mouth) once via keypress, then linearly map between them. No live
    drift, so a single bad frame can't distort the whole range.

    Field names (fist_ratio/open_ratio) are hand-gesture terms but this is
    generic - mouth calibration reuses the same class with set_fist() as
    "mouth closed" and set_open() as "mouth open".
    """

    def __init__(self):
        self.fist_ratio = None
        self.open_ratio = None

    @property
    def is_ready(self):
        return self.fist_ratio is not None and self.open_ratio is not None

    def set_fist(self, ratio):
        self.fist_ratio = ratio

    def set_open(self, ratio):
        self.open_ratio = ratio

    def save(self, path):
        if not self.is_ready:
            return
        path.write_text(json.dumps({
            "fist_ratio": self.fist_ratio,
            "open_ratio": self.open_ratio,
        }))

    def load(self, path):
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self.fist_ratio = data["fist_ratio"]
            self.open_ratio = data["open_ratio"]
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def to_percent(self, ratio):
        if not self.is_ready:
            return None
        span = self.open_ratio - self.fist_ratio
        if abs(span) < 1e-6:
            return None
        percent = (ratio - self.fist_ratio) / span * 100
        return max(0.0, min(100.0, percent))

    def midpoint_threshold(self, hysteresis_fraction=0.15):
        """Returns (low, high, margin) for classifying a ratio as clearly
        below/above the calibrated midpoint, with a dead zone near the
        middle to avoid flicker."""
        lo, hi = self.fist_ratio, self.open_ratio
        if hi < lo:
            lo, hi = hi, lo
        margin = (hi - lo) * hysteresis_fraction
        return lo, hi, margin
