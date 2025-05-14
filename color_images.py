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
#from torch.optim.lr_scheduler import OneCycleLR

import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Munch.fromDict(cfg)



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
                         do_jigsaw = cfg.active_heads.jigsaw, 
                         pretrained = True,
                         do_classification = cfg.active_heads.classification, 
                         do_coloring = cfg.active_heads.coloring, 
                         jigsaw_cfg = cfg.jigsaw_cfg,
                         pixel_shuffle_cfg = cfg.pixel_shuffle_cfg,
                         verbose = cfg.verbose,
                         pretrained_model_info = cfg.pretrained_info) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    base = cfg.pretrained_info.link.split("/checkpoints")[0]
    images_url = f"{base}/colored_images"
    recolor_images(data_path='data', output_dir=images_url, split='val', model=model, n_images=20, shuffle=False, img_size=cfg.img_size)
    return

if __name__ == '__main__':
    main()