from collections import OrderedDict
from torch import Tensor,nn
from .binary_loss import DiceLoss
from .prob_loss import HardBCELoss
from .thresh_loss import MarkL1Loss


class DBLoss(nn.Module):
    def __init__(self, prob_scale:float, thresh_scale:float, binary_scale:float, **kwargs):
        super().__init__()
        self.prob_scale: float = prob_scale
        self.prob_loss: HardBCELoss = HardBCELoss(**kwargs['prob_loss'])
        self.thresh_scale: float = thresh_scale
        self.thresh_loss: MarkL1Loss = MarkL1Loss(**kwargs['thresh_loss'])
        self.binary_scale: float = binary_scale
        self.binary_loss: DiceLoss = DiceLoss(**kwargs['binary_loss'])

    def forward(self, pred, batch) -> tuple:
        prob_dist: Tensor = self.prob_loss(
            pred['prob_map'], batch['prob_map'], batch['prob_mask'])
        loss: Tensor = prob_dist
        loss_list: OrderedDict = OrderedDict(prob_loss=prob_dist)
        if 'thresh_map' in pred:
            thresh_dist: Tensor = self.thresh_loss(
                pred['thresh_map'], batch['thresh_map'], batch['thresh_mask'])
            binary_dist: Tensor = self.binary_loss(
                pred['binary_map'], batch['prob_map'], batch['prob_mask'])
            loss_list.update(thresh_loss=thresh_dist, binary_loss=binary_dist)
            loss = self.binary_scale * binary_dist + self.thresh_scale * \
                thresh_dist + self.prob_scale * prob_dist
        return loss, loss_list
