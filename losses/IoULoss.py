import torch
import torch.nn as nn

from losses.AbstractLoss import AbstractLoss

class IoULoss(AbstractLoss):
    """IoULoss is a loss function used for binary segmentation tasks.

    Args:
        AbstractLoss (ABC, nn.Module): Abstract class for custom loss functions.
    """
    def __init__(self, smooth=0):
        """Initializes the IoULoss module.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero 0. Defaults to 0.
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, output, target):
        y_pred = output.view(-1)
        y_true = target.view(-1)
        
        intersection = (y_pred * y_true).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)