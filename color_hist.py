from dataset_functions.multitask_dataloader import MultiTaskDataset
import numpy as np
from munch import Munch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch

def quantize_ab_channels(ab_tensor, bin_size=0.1, vmin=-1.0, vmax=1.0):
    """
    Quantize ab channels into discrete bins.

    Args:
        ab_tensor: (B, 2, H, W) tensor with values in [-1, 1]
        bin_size: Size of each bin (e.g., 0.1)
        vmin: Minimum value in the range
        vmax: Maximum value in the range

    Returns:
        bin_indices: (B, 2, H, W) LongTensor with bin indices
    """
    num_bins = int((vmax - vmin) / bin_size)
    ab_clamped = ab_tensor.clamp(min=vmin, max=vmax - 1e-6)  # ensure max value fits in last bin

    # Shift and scale to get bin index
    bin_indices = ((ab_clamped - vmin) / bin_size).floor().long()

    return bin_indices

import torch

# Example: ab_bin_indices has shape (B, 2, H, W), with values in [0, 19]
# Assuming ab_bin_indices is from quantize_ab_channels()
def compute_ab_histogram(ab_bin_indices, num_bins=20):
    # Split a and b
    a_bins = ab_bin_indices[:, 0, :, :].reshape(-1)  # shape: (B*H*W,)
    b_bins = ab_bin_indices[:, 1, :, :].reshape(-1)

    # Create 2D histogram
    hist = torch.zeros((num_bins, num_bins), dtype=torch.float32, device=ab_bin_indices.device)

    # Compute 1D flat indices: index = a * num_bins + b
    indices = a_bins * num_bins + b_bins
    flat_hist = torch.bincount(indices, minlength=num_bins * num_bins)

    # Reshape back to 2D
    hist = flat_hist.view(num_bins, num_bins)

    return hist

def main():
    df = MultiTaskDataset('data', split='train', img_size = 224, num_patches=4)

    dl = DataLoader(train_dataset, 
                    batch_size=32, 
                    shuffle=False,
                    num_workers=32)

    hist = torch.zeros((20,20), dtype=np.float32)

    for img,label in dl:
        ab = label.ab_image
        print(ab.shape)
        break



if __name__ == "__main__":
    main()