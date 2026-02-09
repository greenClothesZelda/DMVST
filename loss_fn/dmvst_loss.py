
import torch.nn as nn

class DMVSTLoss(nn.Module):
    def __init__(self, gamma=1.0, eps=0.5, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        term1 = diff ** 2

        relative_diff = diff / (y_true + self.eps)
        term2 = relative_diff ** 2
        loss = term1 + (self.gamma * term2)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss