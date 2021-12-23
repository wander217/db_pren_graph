from torch import nn, Tensor
from typing import Tuple


class AttnLoss(nn.Module):
    def __init__(self, fl_ratio: float, sl_ratio: float, ol_ratio: float, pad: int):
        super().__init__()
        self.f_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=pad)
        self.s_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=pad)
        self.o_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=pad)
        self.fl_ratio: float = fl_ratio
        self.sl_ratio: float = sl_ratio
        self.ol_ratio: float = ol_ratio
        self.pad: int = pad

    def forward(self, f_pred: Tensor, s_fred: Tensor, o_fred: Tensor, target: Tensor) -> Tuple:
        target = target.contiguous().view(-1)
        fl: Tensor = self.f_loss(f_pred.view(-1, f_pred.size(-1)), target)
        sl: Tensor = self.s_loss(s_fred.view(-1, s_fred.size(-1)), target)
        ol: Tensor = self.o_loss(o_fred.view(-1, o_fred.size(-1)), target)
        loss = self.fl_ratio * fl + self.sl_ratio * sl + self.ol_ratio * ol
        return loss, fl, sl, ol
