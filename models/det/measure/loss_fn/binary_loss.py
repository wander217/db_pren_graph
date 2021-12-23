from torch import nn, Tensor


class DiceLoss(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps: float = eps

    def forward(self, pred: Tensor, prob_map: Tensor, prob_mask: Tensor) -> Tensor:
        pred = pred[:, 0, :, :]
        prob_map = prob_map[:, 0, :, :]

        intersection: Tensor = (pred * prob_map * prob_mask).sum()
        uninon: Tensor = (prob_map*prob_mask).sum() + \
            (pred*prob_mask).sum() + self.eps
        loss: Tensor = 1. - 2. * intersection / uninon
        assert loss <= 1., loss
        return loss
