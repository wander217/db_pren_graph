from torch import nn, Tensor
from .backbone import ATTRBackbone
from models.rec.dataset import Alphabet
from typing import Dict


class PRENModel(nn.Module):
    def __init__(self, alphabet: Alphabet, backbone: Dict):
        super().__init__()
        self.backbone: nn.Module = ATTRBackbone(alphabet, **backbone)
        self.apply(self.weight_init)

    def forward(self, image: Tensor):
        """
            :param image: (B,3,H,W)
            :return:
                pred: (B,T,vocab)
                output: (B,T,vocab)
        """
        pred = self.backbone(image)
        return pred

    def weight_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
