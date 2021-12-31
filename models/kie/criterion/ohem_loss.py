import numpy as np
import torch
from torch import nn, Tensor


class OHEMLoss(nn.Module):
    def __init__(self, ohem: int):
        super().__init__()
        self.ohem: int = ohem
        self.loss_fn: nn.Module = nn.CrossEntropyLoss(ignore_index=-100)

    def _ohem(self, pred: Tensor, label: Tensor) -> Tensor:
        pred = pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        pos_num: int = sum(label != 0)
        neg_sum: int = pos_num * self.ohem
        pred_value: np.ndarray = pred[:, 1:].max(1)
        neg_score_sorted: np.ndarray = np.sort(-pred_value[label == 0])
        if neg_score_sorted.shape[0] > neg_sum:
            threshold = -neg_score_sorted[neg_sum - 1]
            mask = ((pred_value >= threshold) | (label != 0))
        else:
            mask = label != -1
        return torch.from_numpy(mask)

    def forward(self, pred: Tensor, label: Tensor) -> Tensor:
        mask_label: Tensor = label.clone()
        mask: Tensor = self._ohem(pred, label).to(pred.device)
        mask_label[mask == False] = -100
        loss: Tensor = self.loss_fn(pred, mask_label)
        return loss
