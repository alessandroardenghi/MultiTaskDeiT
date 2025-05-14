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
from utils import hamming_acc, freeze_components, recolor_images, load_partial_checkpoint
from timm import create_model
from logger import TrainingLogger

import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Munch(cfg)  



def main():
    
    cfg = load_config('config.yaml')        # cfg dict with all attributes inside
    logger = TrainingLogger()
    logger.save_config(cfg, filename='config.yaml')
    
    active_heads = cfg.active_heads
    do_coloring = 'coloring' in active_heads
    do_classification = 'classification' in active_heads
    do_jigsaw = 'jigsaw' in active_heads
    
    if cfg.weights != '':
        weights = torch.from_numpy(np.load(cfg.weights))
    else:
        weights = None
    
    
    model = create_model(cfg.model_name, 
                         do_jigsaw = do_jigsaw, 
                         do_classification = do_classification, 
                         do_coloring = do_coloring, 
                         n_jigsaw_patches = cfg.jigsaw_patches,
                         pixel_shuffle = cfg.pixel_shuffle,
                         verbose = cfg.verbose,
                         pretrained = cfg.pretrained_backbone) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    if cfg.pretrained_checkpoint:
        load_partial_checkpoint(model, cfg.pretrained_checkpoint, cfg.verbose)
    freeze_components(model, component_names=cfg.modules_to_freeze, freeze=True, verbose=cfg.verbose)
    freeze_components(model, component_names=cfg.modules_to_unfreeze, freeze=False, verbose=cfg.verbose)
    

    
    recolor_images(data_path='data', output_dir='coloring_test5', split='val', model=model, n_images=100, shuffle=True)
    return

if __name__ == '__main__':
    main()