class LinearDecay:
    def __init__(self, steps: int, startp: float = 1.0, endp: float = 0.0) -> None:
        self.steps = steps
        self.startp = startp
        self.endp = endp

    def __call__(self, epoch: int) -> float:
        epoch = max(min(epoch, self.steps), 0)
        return self.startp + (self.endp - self.startp) * (epoch / float(self.steps))
