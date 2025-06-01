import numpy as np
import torch
import random
import cv2
import os
import torch.nn as nn
from timm import create_model

################################################################################################
######## DATA UTILS ############################################################################
################################################################################################

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
    
    transformation_vector = [[idx, 0] for idx in range(n**2)]
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
    # if flip:
    #     for idx, patch in enumerate(patches):   
    #         flip = np.random.choice([0, 1, 2])
    #         if flip == 1:  # Horizontal flip
    #             patch = np.fliplr(patch)
    #         elif flip == 2:  # Vertical flip
    #             patch = np.flipud(patch)
    #         patches[idx] = patch
    #         transformation_vector[idx][2] = flip
    
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
    #print(images.shape)
    N = n_patches * n_patches

    transformed_images = torch.empty_like(images)
    pos_vectors = torch.empty((B, N), dtype=torch.long)
    rot_vectors = torch.empty((B, N), dtype=torch.long)

    for i in range(B):
        transformed, pos, rot = jigsaw_single_image(images[i], n_patches)
        transformed_images[i] = transformed
        pos_vectors[i] = pos
        rot_vectors[i] = rot
    # print(pos_vectors.shape)
    # print(rot_vectors.shape)

    return transformed_images, pos_vectors, rot_vectors

################################################################################################
######## METRICS UTILS ########################################################################
################################################################################################

def multilabel_recall(y_pred, y_true, mask=None):
    """
    Compute recall for multilabel classification. It computes the ratio of true positives 
    to total ground truth positives. Useful for evaluating models on multilabel tasks becuase the number
    of ones in a label vector is sparse (over 80 classes there are few objects per image).

    It can be interpreted as the number of objects that are correctly predicted out of all objects 
    that are present in the image.
    Args:
        y_pred: Tensor of shape (B, N), binary predictions (0 or 1).
        y_true: Tensor of shape (B, N), binary ground truth labels (0 or 1).
        mask: Optional mask to select specific classes.
    Returns:
        recall: Recall value (float).
        total_positives: Total number of ground truth positives (int).
    """
    if mask is not None:
        y_pred = y_pred[:,mask]
        y_true = y_true[:,mask]

    # Element-wise AND -> only positions where both pred and label are 1
    correct_positives = ((y_pred == 1) & (y_true == 1)).sum().item()

    # Total ground truth positives (i.e., how many 1s in labels)
    total_positives = (y_true == 1).sum().item()

    return correct_positives / total_positives if total_positives > 0 else 0, total_positives

def multilabel_precision(y_pred, y_true, mask=None):
    """
    Compute precision for multilabel classification. It computes the ratio of true positives
    to total predicted positives.
    Args:
        y_pred: Tensor of shape (B, N), binary predictions (0 or 1).
        y_true: Tensor of shape (B, N), binary ground truth labels (0 or 1).
        mask: Optional mask to select specific classes.
    Returns:
        precision: Precision value (float).
        total_predicted: Total number of predicted positives (int).
    """
    if mask is not None:
        y_pred = y_pred[:,mask]
        y_true = y_true[:,mask]

    # Element-wise AND -> positions where both pred and label are 1
    correct_positives = ((y_pred == 1) & (y_true == 1)).sum().item()

    # Total predicted positives (i.e., how many 1s in predictions)
    total_predicted = (y_pred == 1).sum().item()

    return correct_positives / total_predicted if total_predicted > 0 else 0, total_predicted

def multilabel_f1(prec, recal):
    """
    Compute F1 score from precision and recall.
    Args:
        prec: Precision value
        recal: Recall value
    Returns:
        F1 = 2 * (prec * recal) / (prec + recal)
    """
    if prec + recal == 0:
        return 0
    else:
        return 2 * (prec * recal) / (prec + recal)

def multilabel_accuracy(y_pred, y_true, mask=None):
    """
    Compute accuracy for multilabel classification.
    Args:
        y_pred: Tensor of shape (B, N), binary predictions (0 or 1).
        y_true: Tensor of shape (B, N), binary ground truth labels (0 or 1).
        mask: Optional mask to select specific classes.
    Returns:
        accuracy: Accuracy value (float).
        total_samples: Total number of samples processed.
    """
    if mask is not None:
        y_pred = y_pred[:,mask]
        y_true = y_true[:,mask]
    
    # Compare predictions and labels element-wise -> shape: (B, N)
    matches = (y_pred == y_true)

    # For each vector, check if all elements match -> shape: (B,)
    exact_matches = matches.all(dim=1)
    accuracy = exact_matches.float().mean().item()

    return accuracy, y_pred.shape[0]

def compute_micro_precision_recall_f1(total_tp, total_fp, total_fn):
    """
    Compute micro-averaged precision, recall, and F1 score.
    
    Args:
        total_tp: Total true positives across batches
        total_fp: Total false positives across batches
        total_fn: Total false negatives across batches
    Returns:
        precision, recall, f1_score
    """
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1

def compute_perclass_f1(tp, fp, fn):
    """
    Compute per-class precision, recall, and F1 score.
    Args:
        tp: Tensor of shape (N,) - true positives for each class
        fp: Tensor of shape (N,) - false positives for each class
        fn: Tensor of shape (N,) - false negatives for each class
    Returns:
        precision: List of precision values for each class
        recall: List of recall values for each class
        f1: List of F1 scores for each class
        macro_f1: Macro-averaged F1 score
    """
    epsilon = 1e-8  # for numerical stability

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    macro_f1 = f1.mean().item()
    return precision.tolist(), recall.tolist(), f1.tolist(), macro_f1

def update_perclass_metrics(y_pred, y_true, mask=None):
    """
    Compute true positives, false positives, and false negatives for a batch.
    
    Args:
        y_pred: Tensor of shape (B, N), binary predictions.
        y_true: Tensor of shape (B, N), binary ground truth.

    Returns:
        A tuple of (true_positives, false_positives, false_negatives)
    """
    if mask is not None:
        y_pred = y_pred[:,mask]
        y_true = y_true[:,mask]
    tp = ((y_pred == 1) & (y_true == 1)).sum(dim=0)
    fp = ((y_pred == 1) & (y_true == 0)).sum(dim=0)
    fn = ((y_pred == 0) & (y_true == 1)).sum(dim=0)

    return tp, fp, fn

class AverageMeter:
    """
    Computes and stores the sum, count, average and current value.
    This is useful for tracking metrics during training or evaluation.
    """

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
    """
    Computes the accuracy of jigsaw predictions.
    This class tracks the total number of patches, correct predictions, and top-n accuracy.
    It can be used to evaluate the performance of a model on jigsaw tasks.
    Attributes:
        n (int): The number of top predictions to consider for accuracy. Top-1 accuracy is returned by default always.
                 If n > 1, top-n accuracy is also computed, otherwise only top-1 accuracy is computed.
    """
    def __init__(self, n = 1):
        self.n = n
        self.reset()

    def reset(self):
        self.total_patches = 0
        self.correct = 0
        self.top_n = 0

    def update(self, pred, gt):
        """
        Args:
            pred: Tensor of shape (B, P, C) — logits per row
            gt: Tensor of shape (B, P) — ground truth labels
        """
        B, P, _ = pred.shape
        total = B * P
        self.total_patches += total

        # Top-1 predictions
        pos_top1 = pred.argmax(dim=-1)
        self.correct += (pos_top1 == gt).sum().item()

        # Top-n accuracy
        if self.n > 1:
            sort = pred.argsort(dim=-1, descending=True)
            topn_pred = sort[:, :, :self.n]
            match = (topn_pred == gt.unsqueeze(-1)).any(dim=-1)
            self.top_n += match.sum().item()

    def get_scores(self):
        total = self.total_patches
        try: 
            acc = self.correct / total
            top_n = self.top_n / total
        except:
            acc = 0
            top_n = 0

        if self.n > 1:
            return {
                'accuracy': acc,
                'topn_accuracy': top_n,
            }
        return {
            'accuracy': acc,
        }

################################################################################################
######## MODELS UTILS ##########################################################################
################################################################################################

def freeze_submodule(model, submodule_name, freeze=True):
        """
        Freeze all parameters in a submodule of the model.
        Args:
            model (torch.nn.Module): The model containing the submodule.
            submodule_name (str): The name of the submodule to freeze. Possible submodules:
                - 'patch_embed'
                - 'blocks'
                - 'coloring_decoder'
                - 'jigsaw_head'
                - 'classification_head'
            freeze (bool): If True, freeze the parameters. If False, unfreeze them.
        """

        submodule = dict(model.named_children()).get(submodule_name, None)
        if submodule is None:
            raise ValueError(f"No submodule named '{submodule_name}' found in model.")
        
        for param in submodule.parameters():
            param.requires_grad = not freeze

        if freeze:
            print(f"Froze all parameters in '{submodule_name}'")
        else:
            print(f"Unfroze all parameters in '{submodule_name}'")


def freeze_components(model, component_names, freeze=True, verbose=False):
    """
    Freeze or unfreeze both submodules and named parameters of a model.

    Args:
        model (torch.nn.Module): The model.
        component_names (list[str]): Names of submodules or parameters to freeze/unfreeze.
        freeze (bool): If True, freeze. If False, unfreeze.
    """
    if isinstance(component_names, str):
        component_names = [component_names]

    # Track what was found
    found = []

    # Freeze submodules
    children = dict(model.named_children())
    for name in component_names:
        if name in children:
            for param in children[name].parameters():
                param.requires_grad = not freeze
            found.append(name)

    # Freeze standalone parameters
    parameters = dict(model.named_parameters())
    for name in component_names:
        if name in parameters:
            parameters[name].requires_grad = not freeze
            found.append(name)
    # Report results
    if verbose:
        print('=' * 100)
        print(f"{'FREEZING' if freeze else 'UNFREEZING'} LAYERS")
        for name in component_names:
            if name in found:
                print(f"{'Froze' if freeze else 'Unfroze'} '{name}'")
            else:
                print(f"[Warning] '{name}' not found as submodule or parameter")
        print('='*100)


def load_partial_checkpoint(model, checkpoint_path, verbose=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model_state_dict = model.state_dict()
    
    updated_layers = []
    not_updated_layers = []
    identical_layers = []
    
    for k, v in model_state_dict.items():
        if k in checkpoint_state_dict:
            if model_state_dict[k].shape == checkpoint_state_dict[k].shape:
                if torch.allclose(model_state_dict[k], checkpoint_state_dict[k], atol=1e-6, rtol=1e-5):
                    identical_layers.append(k.split('.')[0])  # Extract block name
                else:
                    updated_layers.append(k.split('.')[0])  # Extract block name
            else:
                not_updated_layers.append(k.split('.')[0])  # Extract block name
        else:
            not_updated_layers.append(k.split('.')[0])  # Extract block name

    # Update only matching layers
    updated_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(updated_state_dict)
    model.load_state_dict(model_state_dict)

    # Remove duplicates from block names
    updated_layers = list(set(updated_layers))
    not_updated_layers = list(set(not_updated_layers))
    identical_layers = list(set(identical_layers))

    # Print results
    if verbose:
        print('='*100)
        print('LAYERS UPDATED FROM LOCAL CHECKPOINT')
        print(f"Updated blocks ({len(updated_layers)}): {updated_layers}")
        print(f"Identical blocks ({len(identical_layers)}): {identical_layers}")
        print(f"Not updated blocks ({len(not_updated_layers)}): {not_updated_layers}")
        print('='*100)
        
def load_pretrained_weights(model, old_model_info, img_size, verbose=False):
    
    if old_model_info.checkpoint_type == 'torch_checkpoint':
        checkpoint = torch.load(old_model_info.link, map_location='cpu')
        source_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    elif old_model_info.checkpoint_type == 'timm_name':
        old_model = create_model(old_model_info.link, 
                                 img_size = img_size,
                                 pretrained=True)
        source_state_dict = old_model.state_dict()
    
    else:
        raise Exception('The provided checkpoint type is not supported')
    
    model_state_dict = model.state_dict()
    
    updated_layers = []
    not_updated_layers = []
    identical_layers = []

    for k, v in model_state_dict.items():
        if k in source_state_dict:

            if model_state_dict[k].shape == source_state_dict[k].shape:
                if torch.allclose(model_state_dict[k], source_state_dict[k], atol=1e-6, rtol=1e-5):
                    identical_layers.append(k.split('.')[0])  # Extract block name
                else:
                    updated_layers.append(k.split('.')[0])  # Extract block name
            else:
                not_updated_layers.append(k.split('.')[0])  # Extract block name
        else:
            not_updated_layers.append(k.split('.')[0])  # Extract block name

    # Update only matching layers
    updated_state_dict = {k: v for k, v in source_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(updated_state_dict)
    model.load_state_dict(model_state_dict) 
    
    # Remove duplicates from block names
    updated_layers = list(set(updated_layers))
    not_updated_layers = list(set(not_updated_layers))
    identical_layers = list(set(identical_layers))

    # Print results
    if verbose:
        print('='*100)
        print('LAYERS UPDATED FROM LOCAL CHECKPOINT')
        print(f"Updated blocks ({len(updated_layers)}): {updated_layers}")
        print(f"Identical blocks ({len(identical_layers)}): {identical_layers}")
        print(f"Not updated blocks ({len(not_updated_layers)}): {not_updated_layers}")
        print('='*100)

def simple_combine_losses(losses, active_heads):
    """ 
    Combine losses based on the active heads.
    alpha : weight for classification loss
    beta : weight for coloring loss
    gamma : weight for jigsaw loss
    Args:
        losses (list): List of losses.
        active_heads (list): List of active heads.
    Returns:
        combined_loss (float): Combined loss.
    """
    alpha = 1.1
    beta = 6.0
    gamma = 0.05
    combined_loss = alpha * losses[0] + beta * losses[1] + gamma * losses[2]
    return combined_loss

################################################################################################
######## RECONSTRUCTION UTILS ##################################################################
################################################################################################

def recolor_image(L, ab):
    # Ensure L and ab are properly formatted
    L_denorm = L[0].numpy() * 255.0  # Shape: [H, W], denormalize L channel
    ab_denorm = (ab.numpy() * 127.0) + 128    # Shape: [2, H, W], denormalize a/b channels
    
    # Ensure L and ab are in valid range
    L_denorm = np.clip(L_denorm, 0, 255)
    ab_denorm = np.clip(ab_denorm, 0, 255)

    # Stack L and ab channels to get LAB image
    # ab_denorm is [2, H, W], we need to transpose it so it becomes [H, W, 2]
    ab_denorm = np.transpose(ab_denorm, (1, 2, 0))  # Shape: [H, W, 2]
    # Stack L channel and ab channels to form LAB image (Shape: [H, W, 3])
    lab_denorm = np.concatenate([L_denorm[..., np.newaxis], ab_denorm], axis=-1)  # Shape: [H, W, 3]
    
    # Convert LAB to RGB using OpenCV
    rgb_reconstructed = cv2.cvtColor(lab_denorm.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Clip the result to ensure it is in the valid range [0, 255]
    rgb_reconstructed = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)

    return rgb_reconstructed
        
def jigsaw_prediction(pred):
    """
    Reorder jigsaw predictions to get unique predictions per each patch position. It uses a greedy algorithm 
    to select the most confident patch predictions for each image, ensuring that each patch is used only once.
    Args:
        pred: Tensor of shape (B, C, C) — softmax probabilities per image per patch
    returns: 
        predictions: Tensor of shape (B, C) — unique predictions per image (reconstructed patch positions)
    """
    softmax = torch.nn.Softmax(dim=-1)
    pred = softmax(pred)
    B, C, _ = pred.shape
    device = pred.device
    predictions = torch.full((B, C), -1, dtype=torch.long, device=device)
    assigned = torch.zeros((B, C), dtype=torch.bool, device=device)

    for i in range(C):
        mask = assigned.unsqueeze(1).expand(-1, C, -1)  # (B, C, C)
        masked_pred = pred.masked_fill(mask, -1) # fill of -1 the values of pred indexed by mask

        # Get max per images: in each image find index of highest value for each patch (row)
        values, indices = masked_pred.max(dim=2)  # (B, C) each entry indices[b, c] is the patch index 
                                                  # with the maximum value in pred[b, c, :]

        # For each image, pick the patch with the highest of these max values
        _, row_indices = values.max(dim=1)  # (B,), (B,) each entry row_indices[b] is the index c 
                                            # of the patch in image b that had the highest max value 
                                            # (most confident prediction among patches that haven’t been used yet)
        col_indices = indices[torch.arange(B), row_indices]  # (B,)

        # Store predictions
        predictions[torch.arange(B), row_indices] = col_indices
        # Update assigned columns
        assigned[torch.arange(B), col_indices] = True
        # Mask out the selected row so it won't be picked again
        pred[torch.arange(B), row_indices, :] = -1

    return predictions

