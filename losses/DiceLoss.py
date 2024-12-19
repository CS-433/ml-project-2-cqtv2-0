import torch
import torch.nn as nn

from losses.AbstractLoss import AbstractLoss
class DiceLoss(AbstractLoss):
    """DiceLoss is a loss function used for binary segmentation tasks.

    Args:
        AbstractLoss (ABC, nn.Module): Abstract class for custom loss functions.
    """
    def __init__(self, smooth=0):
        """
        Initializes the DiceLoss module.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero. Defaults to 0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        """
        Forward pass for DiceLoss.

        Args:
            probs (torch.Tensor): Predicted logits, shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        
        # Flatten the tensors
        probs_flat = probs.flatten()
        targets_flat = targets.flatten()

        # Compute intersection and union
        intersection = torch.sum(probs_flat * targets_flat)
        union = torch.sum(probs_flat**2) + torch.sum(targets_flat**2)
        # Compute Dice coefficient
        dice = (2. * intersection) / (union + self.smooth)
        dice_loss = 1 - dice
        return dice_loss
    