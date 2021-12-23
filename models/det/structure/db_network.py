import numpy as np
from torch import nn, Tensor
from .model import DBModel
from models.det.measure import DBLoss
from typing import Tuple, OrderedDict


class DBNetwork(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.loss_fn = DBLoss(**kwargs['loss_fn'])
        self.loss_fn = nn.DataParallel(self.loss_fn)
        self.model = DBModel(**kwargs['model'])
        self.model = nn.DataParallel(self.model)
        self.device = device

    def forward(self, batch: dict) -> Tuple:
        image: Tensor = batch['image']
        image = image.to(self.device)
        pred: Tensor = self.model(image.float())

        for key, value in batch.items():
            if (value is not None) and hasattr(value, 'to'):
                batch[key] = value.to(self.device)
        loss, metric = self.loss_fn(pred, batch)
        return loss, pred, metric

    def predict(self, image: Tensor):
        return self.model(image)
