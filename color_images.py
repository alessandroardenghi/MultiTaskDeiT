import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import models.model_registry
import timm
import os
import numpy as np
from dataset_functions.multitask_dataloader import MultiTaskDataset
from munch import Munch
from loss import WeightedL1Loss, WeightedMSELoss
from utils import AverageMeter, JigsawAccuracy
from models.full_model import MultiTaskDeiT
from multitask_training import train_model
from timm import create_model
from logger import TrainingLogger
from PIL import Image
#from torch.optim.lr_scheduler import OneCycleLR

import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Munch.fromDict(cfg)

import cv2
import numpy as np

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

def recolor_images(data_path, output_dir, split, model, n_images, img_size, shuffle=False):
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultiTaskDataset(data_path, split=split, img_size=img_size, do_coloring=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    for i, (images, labels) in enumerate(loader):
        print(images.keys())
        if i >= n_images:
            break
        output = model(images)

        # Assuming images.original_image[0] is a torch.Tensor
        colored_img = recolor_image(images.image_colorization[0].detach(), output.pred_coloring[0].detach())
        colored_img = Image.fromarray(colored_img)
        colored_img.save(os.path.join(output_dir,f'image{i}.jpg'))


def main():
    
    cfg = load_config('config.yaml')        # cfg dict with all attributes inside
    logger = TrainingLogger(experiment_name=cfg.experiment_name)
    logger.save_config(cfg, filename='config.yaml')

    
    if cfg.weights != '':
        weights = torch.from_numpy(np.load(cfg.weights))
    else:
        weights = None
    
    
    model = create_model(cfg.model_name, 
                         img_size = cfg.img_size,
                         do_jigsaw = False, 
                         pretrained = True,
                         n_classes = 80,
                         do_classification = False, 
                         do_coloring = True, 
                         jigsaw_cfg = cfg.jigsaw_cfg,
                         pixel_shuffle_cfg = cfg.pixel_shuffle_cfg,
                         verbose = cfg.verbose,
                         pretrained_model_info = cfg.pretrained_info) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    base = 'historic_test'
    images_url = f"{base}/multi"
    recolor_images(data_path='data2', output_dir=images_url, split='test', model=model, n_images=4, shuffle=False, img_size=cfg.img_size)
    return

if __name__ == '__main__':
    main()