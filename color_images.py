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

def recolor_images(data_path, output_dir, split, model, n_images, img_size, shuffle=False):
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultiTaskDataset(data_path, split=split, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    for i, (images, labels) in enumerate(loader):
        if i >= n_images:
            break
        output = model(images)
        #colored_img = recolor_image(images.image_colorization[0], labels.ab_channels[0])
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
    recolor_images(data_path='coco_colors', output_dir=images_url, split='test', model=model, n_images=20, shuffle=False, img_size=cfg.img_size)
    return

if __name__ == '__main__':
    main()