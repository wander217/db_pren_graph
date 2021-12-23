from collections import OrderedDict
from torch import nn, Tensor
from ..backbone import DBEfficientNet
from ..neck import DBNeck
from ..head import DBHead
import torch
import time
import yaml


class DBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = DBEfficientNet(**kwargs['backbone'])
        self.neck = DBNeck(**kwargs['neck'])
        self.head = DBHead(**kwargs['head'])

    def forward(self, x: Tensor):
        brs: list = self.backbone(x)
        nrs: Tensor = self.neck(brs)
        hrs: OrderedDict = self.head(nrs)
        if self.training:
            return hrs
        return hrs['binary_map']


# test
if __name__ == "__main__":
    file_config:str = 'D:\\new_ocr\\core\\config\\det-enet-vndata.yaml'
    with open(file_config) as stream:
        data = yaml.safe_load(stream)
    model = DBModel(**data['structure']['model'])
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total params:', total_params)
    print('train params:', train_params)
    a = torch.rand((1, 3, 800, 800), dtype=torch.float)
    start = time.time()
    b = model(a)
    print('run:', time.time()-start)
