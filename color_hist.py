from dataset_functions.multitask_dataloader import MultiTaskDataset
import numpy as np
from munch import Munch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch
import os
from scipy.ndimage import gaussian_filter


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

def compute_ab_histogram(ab_bin_indices, num_bins=20):
    a_bins = ab_bin_indices[:, 0, :, :].reshape(-1) # new shape: (B*H*W,) -> flattening
    b_bins = ab_bin_indices[:, 1, :, :].reshape(-1) # new shape: (B*H*W,) -> flattening

    # Compute 1D flat indices: index = a * num_bins + b
    indices = a_bins * num_bins + b_bins
    flat_hist = torch.bincount(indices, minlength=num_bins * num_bins)

    # Reshape back to 2D
    hist = flat_hist.view(num_bins, num_bins)

    return hist

def get_weight_matrix(W, Q):

    U = 1.0 / (0.5 * W + 0.5 / Q)
    Z = np.sum(W * U)
    N = U / Z
    return N

def main(weights_path = None):
    lossdir = 'lossweights'
    if not os.path.exists(lossdir):
        os.makedirs(lossdir)

    df = MultiTaskDataset('data', split='train', img_size = 224, do_coloring=True, num_patches=4)
    dl = DataLoader(df, 
                    batch_size=32, 
                    shuffle=False,
                    num_workers=32)

    l = -1
    u = 1
    interval_size = u - l
    num_bins = 25
    bin_size = interval_size / num_bins

    if weights_path is None:
        global_hist = torch.zeros((num_bins, num_bins), dtype=torch.float32)
        for img,label in dl:
            ab = label.ab_channels
            ab_bins = quantize_ab_channels(ab, bin_size=bin_size, vmin=l, vmax=u)
            
            hist = compute_ab_histogram(ab_bins, num_bins=num_bins)
            global_hist += hist

        global_hist_np = global_hist.cpu().numpy()
        smoothed_np = gaussian_filter(global_hist_np, sigma=3.0)  # Try sigma=1.0 or 2.0

        np.save(os.path.join(lossdir, 'hist_25_bins.npy'), global_hist_np)
        #np.save(os.path.join(lossdir, 'smoothed_hist.npy'), smoothed_np)

    else:
        global_hist_np = np.load(weights_path)
    
    p = global_hist_np/np.sum(global_hist_np) #normalize
    p_tilde = gaussian_filter(p, sigma=5.0)
    Q = np.count_nonzero(global_hist_np)
    w = get_weight_matrix(p_tilde, Q)
    np.save(os.path.join(lossdir, 'weights_25_bins_sigma_3.npy'), w)
        
if __name__ == "__main__":
    main()