from torch import nn,Tensor
import torch
import torch.nn.functional as f

"""""
    1. prediction : N,1,H,W
    2. prob_map : N,1,H,W
    2. prob_mask: N,H,W
"""""
class HardBCELoss(nn.Module):
    def __init__(self, ratio:float, eps:float):
        super().__init__()
        self.ratio:float = ratio
        self.eps:float = eps

    def forward(self, pred:Tensor, prob_map:Tensor, prob_mask:Tensor)->Tensor:
        pos:Tensor = (prob_map*prob_mask).byte()
        neg:Tensor = ((1-prob_map)*prob_mask).byte()

        pos_num:int = int(pos.float().sum())
        neg_num:int = min(int(neg.float().sum()), int(pos_num*self.ratio))
        loss:Tensor = f.binary_cross_entropy(
            pred, prob_map, reduction='none')[:, 0, :, :]

        pos_loss:Tensor = loss * pos.float()
        neg_loss:Tensor = loss * neg.float()
        neg_loss, _ = torch.topk(neg_loss.view(-1), neg_num)
        bce_loss:Tensor = (pos_loss.sum() + neg_loss.sum()) / (pos_num + neg_num + self.eps)
        return bce_loss
