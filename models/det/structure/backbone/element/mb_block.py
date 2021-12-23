from typing import Callable, List
from torch import nn, Tensor
from .se_block import SEBlock
from .bneck_config import BNeckConfig, adjust_size
from torchvision.ops.misc import ConvNormActivation
from torchvision.ops import StochasticDepth


class MBConv(nn.Module):
    def __init__(self, config: BNeckConfig,
                 stochastic_depth_prob: float,
                 norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SEBlock,
                 use_se: bool = True) -> None:
        super().__init__()
        # assert config.stride in (1, 2), config.stride
        self.use_res = (config.stride == 1) and (
            config.input_channel == config.output_channel)
        blocks: List = []
        exp_channel: int = adjust_size(
            config.input_channel, config.expand_ratio)
        if exp_channel != config.input_channel:
            blocks.append(
                ConvNormActivation(
                    in_channels=config.input_channel,
                    out_channels=exp_channel,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=nn.SiLU
                )
            )
        # deepthwise
        blocks.append(
            ConvNormActivation(
                in_channels=exp_channel,
                out_channels=exp_channel,
                kernel_size=config.kernel_size,
                stride=config.stride,
                groups=exp_channel,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU
            )
        )
        # SE
        if use_se:
            blocks.append(se_layer(input_channel=exp_channel))
        # project
        blocks.append(
            ConvNormActivation(
                in_channels=exp_channel,
                out_channels=config.output_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU
            )
        )
        self.layers = nn.Sequential(*blocks)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.output_channel = config.output_channel

    def forward(self, input: Tensor) -> Tensor:
        output = self.layers(input)
        if self.use_res:
            output = self.stochastic_depth(output)
            output += input
        return output
