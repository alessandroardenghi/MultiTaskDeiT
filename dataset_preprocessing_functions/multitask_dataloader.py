import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from utils import jigsaw_single_image
from munch import Munch
import cv2
import numpy as np

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



def preprocess_for_coloring(pil_img):
    """
    Converts a PIL RGB image to normalized L and ab tensors using OpenCV's CIELAB conversion.

    Returns:
        L_3ch_tensor: torch.FloatTensor [3, H, W] - L channel repeated 3 times
        ab_tensor: torch.FloatTensor [2, H, W] - normalized a/b channels in [-1, 1]
    """
    # Convert PIL image to RGB numpy array
    rgb_np = np.array(pil_img.convert("RGB"))  # Shape: (H, W, 3), dtype=uint8

    # Convert RGB to LAB using OpenCV
    lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB).astype(np.float32)  

    L = lab[:, :, 0] / 255.0                   # Normalize L to [0, 1]
    ab = (lab[:, :, 1:] - 128)/ 127.0                 # Normalize ab to roughly [-1, 1]

    # Convert to PyTorch tensors
    L_tensor = torch.from_numpy(L).unsqueeze(0)               # Shape: [1, H, W]
    ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1))       # Shape: [2, H, W]

    # Optional: repeat L channel 3x if needed as model input
    L_3ch_tensor = L_tensor.repeat(3, 1, 1)                    # Shape: [3, H, W]

    return L_3ch_tensor.float(), ab_tensor.float()


import torch
import torch.nn.functional as F

def jigsaw_batch_with_rotation(
    x: torch.Tensor,
    n: int,
    do_rotate: bool = True
):
    """
    Split each image in the batch into n x n patches, randomly shuffle
    those patches, and optionally apply a random 0°/90°/180°/270° rotation
    to each one, then reassemble.

    Args:
        x:         Tensor of shape (B, C, H, W)
        n:         number of patches per spatial dimension
        do_rotate: if True, apply random rotations and return rots;
                   if False, skip rotations and return zeros.

    Returns:
        jigsawed: Tensor (B, C, H, W) with shuffled (± rotated) patches
        perms:    LongTensor (B, n*n) of shuffle indices
        rots:     LongTensor (B, n*n) of rotation codes
                  (0 if no rotation or 0°,
                   1 → 90°, 2 → 180°, 3 → 270°)
    """
    B, C, H, W = x.shape
    ps = H // n
    L = n * n

    # 1) Extract patches → (B, L, C, ps, ps)
    patches = F.unfold(x, kernel_size=ps, stride=ps)
    patches = patches.transpose(1, 2).view(B, L, C, ps, ps)

    # 2) Shuffle
    perms = torch.argsort(torch.rand(B, L, device=x.device), dim=1)
    flat = patches.view(B, L, C * ps * ps)
    idx = perms.unsqueeze(-1).expand(-1, -1, C * ps * ps)
    shuffled = flat.gather(1, idx).view(B, L, C, ps, ps)

    # 3) Rotation (optional)
    if do_rotate:
        # random 0–3 per patch
        rots = torch.randint(0, 4, (B, L), device=x.device)
        sp  = shuffled.view(-1, C, ps, ps)   # (B*L, C, ps, ps)
        kr  = rots.view(-1)                  # (B*L,)
        out = torch.empty_like(sp)

        for r in range(4):
            idxs = (kr == r).nonzero(as_tuple=True)[0]  # 1D indices
            if idxs.numel() > 0:
                out[idxs] = torch.rot90(sp[idxs], k=r, dims=(2,3))

        rotated = out.view(B, L, C, ps, ps)
    else:
        # no rotation: rots all zero, pass-through
        rots    = torch.zeros(B, L, dtype=torch.long, device=x.device)
        rotated = shuffled

    # 4) Fold back → (B, C, H, W)
    flat2    = rotated.view(B, L, C * ps * ps).transpose(1, 2)
    jigsawed = F.fold(flat2, output_size=(H, W), kernel_size=ps, stride=ps)

    return jigsawed, perms, rots

def jigsaw_single(x: torch.Tensor, n: int, do_rotate=False):
    # x: (C, H, W)
    x_batched = x.unsqueeze(0)                      # → (1, C, H, W)
    jigs, perms, rots = jigsaw_batch_with_rotation(x_batched, n, do_rotate=do_rotate)        # your fast batched fn
    return jigs.squeeze(0), perms.squeeze(0), rots.squeeze(0) 

class MultiTaskDataset(Dataset):
    
    def __init__(self, 
                 data_dir, 
                 img_size = 224, 
                 split='train', 
                 num_patches = 14, 
                 transform = True,
                 do_jigsaw = False,
                 do_coloring = False,
                 do_classification = False,
                 do_rotate=False, 
                 weights=None):
        
        if split not in ['train', 'val', 'test']:
            raise Exception('Split must be chosen between train, test and val.')
        
        self.weights = weights
        self.do_coloring = do_coloring
        self.do_classification = do_classification
        self.do_jigsaw = do_jigsaw
        self.img_size = img_size
        self.do_rotate=do_rotate
        self.transform = transform
        self.image_dir = os.path.join(data_dir, 'images')
        #npz_data = np.load(os.path.join(data_dir, 'labels.npz'), allow_pickle=True)
        #self.labels_dict = npz_data['labels'].item()
        self.num_patches = num_patches
        
        if transform and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2,   # Adjust brightness by a large amount
                                        contrast=0.1,     # Adjust contrast for deeper darks and bright highlights
                                        saturation=0.1,   # Increase saturation to make colors more vivid
                                        hue=0.05),
                #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size))
            ])
            

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        #image = image.resize((self.img_size, self.img_size))
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = image.resize((self.img_size, self.img_size))
            
            
        images = Munch()
        labels = Munch()
        
            
        image_grayscale, ab_channels = preprocess_for_coloring(image)
        
        if self.do_jigsaw:
            image_jigsaw, pos_vec, rot_vec = jigsaw_single(image_grayscale, n = self.num_patches, do_rotate=self.do_rotate)
            images.image_jigsaw = image_jigsaw
            labels.pos_vec = pos_vec
            labels.rot_vec = rot_vec
        
            
        if self.do_classification:
            if image_name not in self.labels_dict.keys():
                
                raise Exception(f'{image_name} LABEL NOT FOUND')
            
            label_classification = self.labels_dict[image_name]
            label_classification = torch.tensor(label_classification, dtype=torch.float32)
            images.image_classification = image_grayscale
            labels.label_classification = label_classification
        
        
        if self.do_coloring:
            
            images.image_colorization = image_grayscale
            labels.ab_channels = ab_channels
            
            
            if self.weights is not None:
                idxs = quantize_ab_channels(ab_channels)
                a_idx = idxs[0]    # shape (H, W)
                b_idx = idxs[1]    # shape (H, W)

                weight_tensor = self.weights[a_idx, b_idx]    # shape: (H, W)
                #weight_tensor = weight_tensor.unsqueeze(0)
                labels.weight_tensor = weight_tensor
        images.original_image = transforms.ToTensor()(image)
        return images, labels