class Holder:
    def __init__(self):
        self.value: float = 0
        self.step: int = 0

    def update(self, value: float, step: int):
        self.value += value
        self.step += step

    def average(self):
        if self.step == 0:
            return 0
        return self.value / self.step