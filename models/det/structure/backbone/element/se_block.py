from torch import nn, Tensor


class SEBlock(nn.Module):
    def __init__(self, input_channel: int, reduction: int = 4, bias: bool = False):
        super().__init__()
        exp: int = max(input_channel//reduction, 1)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channel, exp, kernel_size=1, bias=bias),
            nn.SiLU(inplace=True),
            nn.Conv2d(exp, input_channel, kernel_size=1, bias=bias),
            nn.Hardsigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        scale = self.block(input)
        return scale * input
