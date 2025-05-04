import numpy as np
import torch
import random
from much import Munch

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

def jigsaw_single_image(image: torch.Tensor, n_patches: int = 14):
    """
    image: torch.Tensor of shape [C, H, W]
    returns:
        transformed_image: torch.Tensor [C, H, W]
        pos_vector: torch.Tensor [N]
        rot_vector: torch.Tensor [N]
    """
    C, H, W = image.shape
    patch_size = H // n_patches
    num_patches = n_patches * n_patches

    patches = []
    pos_vector = torch.empty(num_patches, dtype=torch.long)
    rot_vector = torch.empty(num_patches, dtype=torch.long)

    # 1. Extract patches
    for i in range(n_patches):
        for j in range(n_patches):
            patch = image[:, 
                          i * patch_size:(i + 1) * patch_size, 
                          j * patch_size:(j + 1) * patch_size]
            patches.append(patch)

    # 2. Shuffle
    permuted_indices = torch.randperm(num_patches)
    patches = [patches[i] for i in permuted_indices]
    pos_vector[:] = permuted_indices

    # 3. Rotate each patch randomly
    for idx, patch in enumerate(patches):
        k = random.choice([0, 1, 2, 3])
        patches[idx] = torch.rot90(patch, k=k, dims=[1, 2])
        rot_vector[idx] = k

    # 4. Reconstruct the image
    transformed_image = torch.zeros_like(image)
    for idx, patch in enumerate(patches):
        i, j = divmod(idx, n_patches)
        transformed_image[:, 
                          i * patch_size:(i + 1) * patch_size, 
                          j * patch_size:(j + 1) * patch_size] = patch

    return transformed_image, pos_vector, rot_vector


def jigsaw_batch(images: torch.Tensor, n_patches: int = 14):
    """
    images: torch.Tensor of shape [B, C, H, W]
    Returns:
        transformed_images: torch.Tensor [B, C, H, W]
        pos_vectors: torch.Tensor [B, N]
        rot_vectors: torch.Tensor [B, N]
    """
    B, C, H, W = images.shape
    print(images.shape)
    N = n_patches * n_patches

    transformed_images = torch.empty_like(images)
    pos_vectors = torch.empty((B, N), dtype=torch.long)
    rot_vectors = torch.empty((B, N), dtype=torch.long)

    for i in range(B):
        transformed, pos, rot = jigsaw_single_image(images[i], n_patches)
        transformed_images[i] = transformed
        pos_vectors[i] = pos
        rot_vectors[i] = rot
    print(pos_vectors.shape)
    print(rot_vectors.shape)

    return transformed_images, pos_vectors, rot_vectors


### TRAINING UTILS ###


def hamming_loss(y_true, y_pred):
    """
    Compute the Hamming loss for multi-label classification. Measures the fraction 
    of correctly predicted labels to the total number of labels.

    Parameters:
    - y_true (Tensor): Ground truth labels of shape [batch_size, num_labels].
    - y_pred (Tensor): Predicted probabilities (after sigmoid) of shape [batch_size, num_labels].
    - threshold (float): The threshold to convert probabilities to binary predictions (default 0.5).

    Returns:
    - hamming_loss (float): The Hamming loss for the batch.
    """

    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    
    correct = np.sum(y_true == y_pred)
    total = np.prod(y_true.shape)
    
    return correct / total

    # if isinstance(y_true, torch.Tensor):
    #     correct = (y_true == y_pred).float().sum()
    #     total = torch.numel(y_true)
    #     return (correct / total).item()

    # else:  # assume numpy
    #     correct = (y_true == y_pred).sum()
    #     total = y_true.size
    #     return correct / total


def save_model(model, path=None):
    """
    Save the model to a specified path.
    Parameters:
    - model (nn.Module): The model to be saved.
    - path (str): Path to save the model.
    Returns:
    - None
    """
    
    name = model.__class__.__name__
    if path is None:
        path = f"{name}.pth"
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + f"/{name}.pth"

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_arch, model_name, path=None):
    """
    Load the model from a specified path.
    Parameters:
    - model_arch (nn.Module): The architecture of the model.
    - model (str): The model to be loaded.
    - path (str): Path to load the model from.
    Returns:
    - model (nn.Module): The loaded model.
    """
    
    if path is None:
        path = f"{model_name}.pth"
    else:
        path = path + f"/{model_name}.pth"
    
    model.load_state_dict(torch.load(path, weight_only=True))
    print(f"Model loaded from {path}")
    return model


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # The current value
        self.avg = 0  # The running average
        self.sum = 0  # The total sum of values
        self.count = 0  # The number of values processed

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # Sum of values
        self.count += n  # Count of samples processed
        self.avg = self.sum / self.count  # Running average

class JigsawAccuracy:
    def __init__(self, n = 3):
        #self.num_classes = num_classes
        self.n = n
        self.reset()

    def reset(self):
        self.total_patches = 0
        self.correct = 0
        self.top_n = 0

    def update(self, pred, gt):
        """
        Args:
            pred: (B, P, C_pos) torch.Tensor
            rotation_logits: (B, P, C_rot) torch.Tensor
            true_positions: (B, P) torch.Tensor
            true_rotations: (B, P) torch.Tensor
        """
        B, P, _ = pred.shape
        total = B * P
        self.total_patches += total

        # Top-1 predictions
        pos_top1 = pred.argmax(dim=-1)
        self.correct += (pos_top1 == gt).sum().item()

        # Top-n accuracy
        if self.n > 1:
            sorted = pred.argsort(dim=-1, descending=True)
            topn_pred = sorted[:, :, :self.n]
            match = (topn_pred == gt.unsqueeze(-1)).any(dim=-1)
            self.top_n += match.sum()


    def get_scores(self):
        total = self.total_patches
        acc = self.correct / total
        top_n = self.top_n / total

        if self.n > 1:
            return {
                'accuracy': acc,
                'topn_accuracy': top_n,
            }
        return {
            'accuracy': acc,
        }


