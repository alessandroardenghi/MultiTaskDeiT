import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        """
        Compute the weighted mean squared error loss.
        Args:
            input (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            weight (torch.Tensor): Weights for each element.
        Shapes: 
            input: (B, 2, image_size, image_size)
            target: (B, 2, image_size, image_size)
            weight: (B, 1, image_size, image_size)
        Returns:
            torch.Tensor: Computed loss.
        """
        weight = weight.unsqueeze(1)           # (B, 1, image_size, image_size)
        weight = weight.expand(-1, 2, -1, -1)  # (B, 2, image_size, image_size)
        error = weight * (input - target) ** 2
        if self.reduction == 'mean':
            return torch.mean(error)
        elif self.reduction == 'sum':
            return torch.sum(error)
        else:
            raise ValueError("Invalid reduction method. Choose 'mean' or 'sum'.")


class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, input, target, weight):
        """
        Compute the weighted L1 loss.
        Args:
            input (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
            weight (torch.Tensor): Weights for each element.
        Shapes: 
            input: (B, 2, image_size, image_size)
            target: (B, 2, image_size, image_size)
            weight: (B, 1, image_size, image_size)
        Returns:
            torch.Tensor: Computed loss.
        """
        weight = weight.unsqueeze(1)           # (B, 1, image_size, image_size)
        weight = weight.expand(-1, 2, -1, -1)  # (B, 2, image_size, image_size)
        error = weight * torch.abs(input - target)
        if self.reduction == 'mean':
            return torch.mean(error)
        elif self.reduction == 'sum':
            return torch.sum(error)
        else:
            raise ValueError("Invalid reduction method. Choose 'mean' or 'sum'.")


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, input, target):
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=-1, keepdim=False).mean()
