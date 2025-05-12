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
    
    def __init__(self, data_dir, img_size = 224, split='train', transform_classification=None, num_patches = 14, do_rotate=False):
        if split not in ['train', 'val', 'test']:
            raise Exception('Split must be chosen between train, test and val.')
        
        self.img_size = img_size
        self.do_rotate=do_rotate
        self.image_dir = os.path.join(data_dir, 'images')
        npz_data = np.load(os.path.join(data_dir, 'labels.npz'), allow_pickle=True)
        self.labels_dict = npz_data['labels'].item()
        self.num_patches = num_patches

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        self.transform_classification = transform_classification if transform_classification is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        
        #image_classification = self.transform_classification(image)
    
        image_colorization, ab_channels = preprocess_for_coloring(image)
        #image_classification = image_colorization.clone()
        image_jigsaw, pos_vec, rot_vec = jigsaw_single(image_colorization, n = self.num_patches, do_rotate=self.do_rotate)
        #image_jigsaw, pos_vec, rot_vec = image_colorization.clone(), image_colorization.clone(), image_colorization.clone()
    
        if image_name not in self.labels_dict.keys():
            raise Exception('LABEL NOT FOUND')
            
        label_classification = self.labels_dict[image_name]
        label_classification = torch.tensor(label_classification, dtype=torch.float32)
        
        images = Munch(image_classification=image_colorization,
                        image_colorization=image_colorization,
                        image_jigsaw=image_jigsaw)
        labels = Munch(label_classification=label_classification,
                        ab_channels=ab_channels,
                        pos_vec=pos_vec,
                        rot_vec=rot_vec)

        return images, labels