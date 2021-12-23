from torch import nn, Tensor
import torch


class MarkL1Loss(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps: float = eps

    def forward(self, pred: Tensor, thresh_map: Tensor, thresh_mask: Tensor) -> Tensor:
        ele_number: Tensor = thresh_mask.sum()
        if ele_number == 0:
            return ele_number
        loss: Tensor = (torch.abs(pred[:, 0]-thresh_map)
                        * thresh_mask).sum() / (ele_number + self.eps)
        return loss
