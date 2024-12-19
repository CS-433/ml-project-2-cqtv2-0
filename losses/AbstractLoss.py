import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractLoss(nn.Module, ABC):
    """This class is used to create an abstract loss function, 
    this class is used to create our own loss functions and adding 
    them with new loss function without the need to create a new class.

    Args:
        nn (nn): Pytorch nn module
        ABC (ABC): Abstract Base Class
    """
    
    def __init__(self):
        super(AbstractLoss, self).__init__()
        
    @abstractmethod
    def forward(self, probs, targets):
        """This function is used to calculate the loss function.

        Args:
            probs (torch.tensor): The predicted values.
            targets (torch.tensor): The target values.
        """
        pass
    
    def __add__(self, other: nn.Module):
        """This function is used to add two loss functions together.

        Args:
            other (nn.Module): The other loss function to be added.

        Returns:
            AbstractLoss: The new loss function.
        """
        class new_loss(AbstractLoss):
            def __init__(self2):
                super(new_loss, self2).__init__()
                
            def forward(self2, probs, targets):
                return self.forward(probs, targets) + other.forward(probs, targets)
        return new_loss()

    def __mul__(self, scalar: float):
        """This function is used to multiply a loss function by a scalar

        Args:
            scalar (float): The scalar to be multiplied by the loss function.

        Returns:
            AbstractLoss: The new loss function.
        """
        class new_loss(AbstractLoss):
            def __init__(self2):
                super(new_loss, self2).__init__()
                
            def forward(self, probs, targets):
                return scalar * self.forward(probs, targets)
        return new_loss()
    
    