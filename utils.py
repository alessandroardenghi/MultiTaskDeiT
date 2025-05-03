import numpy as np
import torch
def jigsaw_image(image : np.array, 
                 n: int, 
                 jigsaw : bool = False, 
                 rotation : bool =False, 
                 flip : bool =False,
                 noise_mean: int = 0,
                 noise_std : int = 0):

    patch_size = image.shape[0] // n
    if image.shape[0] % n != 0:
        raise ValueError("Image dimensions must be divisible by n.")
    
    transformation_vector = [[idx, 0, 0] for idx in range(n**2)]
    # idx indicates what is the index of the current patch in the original image
    # the second entry indicates the rotation applied to the patch (0, 90, 180, 270)
    # the third entry indicates the flip applied (0 = no flip, 1 = horizontal flip, 2 = vertical flip)
    
    
    # ADDING NOISE 
    image_float = image.astype(np.float32)
    noise = np.random.normal(noise_mean, noise_std, image.shape)
    image_float += noise
    image_float = np.clip(image_float, 0, 255)
    image = image_float.astype(np.uint8)
    
    patches = []
    for i in range(n):
        for j in range(n):
            patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    # APPLYING JIGSAW
    if jigsaw:
        permuted_indices = np.random.permutation(len(patches))
        patches = [patches[i] for i in permuted_indices]
        for i in range(len(transformation_vector)):
            transformation_vector[i][0] = permuted_indices[i]

    # APPLYING ROTATION
    if rotation:
        for idx, patch in enumerate(patches):    
            rot_idx = np.random.choice([0, 1, 2, 3])  
            patches[idx] = np.rot90(patch, k=rot_idx) 
            transformation_vector[idx][1] = rot_idx
    
    # APPLYING FLIP
    if flip:
        for idx, patch in enumerate(patches):   
            flip = np.random.choice([0, 1, 2])
            if flip == 1:  # Horizontal flip
                patch = np.fliplr(patch)
            elif flip == 2:  # Vertical flip
                patch = np.flipud(patch)
            patches[idx] = patch
            transformation_vector[idx][2] = flip
    
    # RECONSTRUCTING IMAGE
    transformed_image = np.zeros_like(image)
    for idx, patch in enumerate(patches):
        i, j = divmod(idx, n)
        transformed_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patch

    return [int(item) for sublist in transformation_vector for item in sublist], transformed_image


def grayscale_weighted_3ch(x):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device).view(1, 3, 1, 1)
    gray = (x * weights).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    return gray.repeat(1, 3, 1, 1)  # (1, 3, H, W)

def add_gaussian_noise(x, mean=0.0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    return torch.clamp(x + noise, 0.0, 1.0)