class LinearDecay:
    def __init__(self, steps: int, startp: int = 1, endp: int = 0) -> None:
        self.steps = steps
        self.startp = startp
        self.endp = endp

    def __call__(self, epoch: int) -> float:
        return self.startp + (self.endp - self.startp) * (epoch / float(self.steps))
