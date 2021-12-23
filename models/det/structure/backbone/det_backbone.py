import math
import copy
import torch
import time
from typing import List
from torch import nn, Tensor
from torchvision.ops.misc import ConvNormActivation
from .element import MBConv, BNeckConfig

# Setting từ bài báo
# width_mul,depth_mul,dropout
NETWORK_SETTING: dict = {
    'b0': (1.0, 1.0, 0.2),
    'b1': (1.0, 1.1, 0.2),
    'b2': (1.1, 1.2, 0.3),
    'b3': (1.2, 1.4, 0.3),
}

DATAPOINT: dict = {
    'b0': [2, 3, 5, 8],
    'b3': [2, 3, 5, 8],
}


class DBEfficientNet(nn.Module):
    def __init__(self, type: str, stochastic_depth_prob: float, use_se: float):
        super().__init__()
        # Lấy setting từ trên
        w_mul, d_mul, _ = NETWORK_SETTING[type]
        self.data_point: list = DATAPOINT[type]
        self.network_setting = [  # resolution ouput b3
            BNeckConfig(1, 3, 1, 32, 16, 1, w_mul, d_mul),  # (1): w/2 * h/2
            BNeckConfig(6, 3, 2, 16, 24, 2, w_mul, d_mul),  # (2): w/4 *h/4 -> Chọn
            BNeckConfig(6, 5, 2, 24, 40, 2, w_mul, d_mul),  # (3): w/8 *h/8 -> Chọn
            BNeckConfig(6, 3, 2, 40, 80, 3, w_mul, d_mul),  # (4): w/16 * h/16
            BNeckConfig(6, 5, 1, 80, 112, 3, w_mul, d_mul),  # (5): w/16 * h/16 -> Chọn
            BNeckConfig(6, 5, 2, 112, 192, 4, w_mul, d_mul),  # (6): w/32 * h/32
            BNeckConfig(6, 3, 1, 192, 320, 1, w_mul, d_mul),  # (7): w/32 * h/32
        ]
        first_output_channel: int = self.network_setting[0].input_channel
        self.layers: nn.ModuleList = nn.ModuleList([
            ConvNormActivation(in_channels=3,
                               out_channels=first_output_channel,
                               kernel_size=3,
                               stride=2,
                               norm_layer=nn.BatchNorm2d,
                               activation_layer=nn.SiLU)
        ])  # (0): w/2 * h/2
        stage_block_id: int = 0
        total_stage_block: int = sum([setting.layer_num for setting in self.network_setting])
        for i, setting in enumerate(self.network_setting):
            stage: List[nn.Module] = []
            se_setting: bool = use_se
            if i == 5 or i == 6:
                se_setting = True
            for _ in range(setting.layer_num):
                block_setting = copy.copy(setting)
                if stage:
                    block_setting.input_channel = block_setting.output_channel
                    block_setting.stride = 1
                new_stochastic: float = stochastic_depth_prob * \
                                        float(stage_block_id) / total_stage_block
                stage.append(MBConv(block_setting, new_stochastic, nn.BatchNorm2d, use_se=se_setting))
                stage_block_id += 1
            self.layers.append(nn.Sequential(*stage))
        # Xây dựng phần cuối
        last_input_channel: int = self.network_setting[-1].output_channel
        last_ouput_channel: int = 4 * last_input_channel
        self.layers.append(
            ConvNormActivation(
                in_channels=last_input_channel,
                out_channels=last_ouput_channel,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.SiLU
            )
        )  # (8): w/32 * h/32 ->Chọn

        # Cài đặt giá trị ban đầu
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init_range = 1 / math.sqrt(module.out_features)
                nn.init.uniform_(module.weight, -init_range, init_range)
                nn.init.zeros_(module.bias)

    def forward(self, input: Tensor) -> List[Tensor]:
        output_list: list[Tensor] = []
        for i in range(len(self.layers)):
            input = self.layers[i](input)
            if i in self.data_point:
                output_list.append(input)
        return output_list


# test
if __name__ == "__main__":
    model = DBEfficientNet("b3", stochastic_depth_prob=0.2, use_se=False)
    a = torch.rand((1, 3, 320, 320), dtype=torch.float)
    start = time.time()
    b = model(a)
    for item in b:
        print(item.size())
    print(b.size())
