import math


def adjust_depth(layer_num: int, depth_mul: float):
    return int(math.ceil(layer_num * depth_mul))


def adjust_size(input_channel: int, expand_ratio: float, divisor: int = 8):
    new_input_channel = max(divisor, int(
        input_channel * expand_ratio + divisor/2) // divisor * divisor)
    if new_input_channel < 0.9 * input_channel:
        new_input_channel += divisor
    return new_input_channel


class BNeckConfig:
    def __init__(self, expand_ratio: float, kernel_size: int,
                 stride: int, input_channel: int,
                 output_channel: int, layer_num: int,
                 width_mul: float, depth_mul: float) -> None:
        self.expand_ratio: float = expand_ratio
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.width_mul: float = width_mul
        self.depth_mul: float = depth_mul
        self.output_channel: int = adjust_size(output_channel, width_mul)
        self.input_channel: int = adjust_size(input_channel, width_mul)
        self.layer_num: int = adjust_depth(layer_num, depth_mul)
