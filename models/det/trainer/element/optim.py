from torch import optim
from typing import Iterator

class Optimizer:
    def __init__(self, name: str, argument: dict):
        self.optimizer = getattr(optim, name)
        self.argument = argument

    def build(self, parameters: Iterator):
        return self.optimizer(
            params=parameters,
            **self.argument
        )