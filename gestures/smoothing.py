class Smoother:
    """Exponential moving average to kill frame-to-frame landmark jitter
    before it reaches the calibration/output stage."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += (x - self.value) * self.alpha
        return self.value
