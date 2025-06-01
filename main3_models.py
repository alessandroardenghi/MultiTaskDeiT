import os
import json
import numpy as np
from munch import Munch
import timm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataset_functions.multitask_dataloader import MultiTaskDataset

import torch.optim as optim
from timm import create_model
import models.model_registry
from models.full_model import MultiTaskDeiT

from loss import WeightedL1Loss, WeightedMSELoss
from multitask_training import train_model
from logger import TrainingLogger
from utils import *


def main():
    
    cfg = load_config('config3_models.yaml')        # cfg dict with all attributes inside
    logger = TrainingLogger(experiment_name=cfg.experiment_name)
    logger.save_config(cfg, filename='config3_models.yaml')

    logger.log(f'loss hyperparameters: alpha=1.1, beta=6, gamma=0.05')
    
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
                         n_classes = cfg.classification_cfg.n_classes,
                         pixel_shuffle_cfg = cfg.pixel_shuffle_cfg,
                         verbose = cfg.verbose,
                         pretrained_model_info1 = cfg.pretrained_info1,
                         pretrained_model_info2 = cfg.pretrained_info2,
                         pretrained_model_info3 = cfg.pretrained_info3) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    
    freeze_components(model, component_names=[module for module, v in cfg.freeze_modules.items() if v], freeze=True, verbose=cfg.verbose)
    freeze_components(model, component_names=[module for module, v in cfg.unfreeze_modules.items() if v], freeze=False, verbose=cfg.verbose)
    
    
    train_dataset = MultiTaskDataset(cfg.data_path, 
                                     split='train', 
                                     img_size = cfg.img_size, 
                                     num_patches=cfg.jigsaw_cfg.n_jigsaw_patches, 
                                     do_rotate=True,
                                     do_jigsaw=cfg.active_heads.jigsaw,
                                     do_coloring=cfg.active_heads.coloring,
                                     do_classification=cfg.active_heads.classification,
                                     weights=weights,
                                     transform=True)
    
    val_dataset = MultiTaskDataset(cfg.data_path, 
                                    split='val', 
                                    img_size = cfg.img_size, 
                                    num_patches=cfg.jigsaw_cfg.n_jigsaw_patches, 
                                    do_rotate=True,
                                    do_jigsaw=cfg.active_heads.jigsaw,
                                    do_coloring=cfg.active_heads.coloring,
                                    do_classification=cfg.active_heads.classification,
                                    weights=weights, 
                                    transform=True)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True,
                                  num_workers=cfg.n_workers)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False,
                                num_workers=cfg.n_workers)
    criterion = Munch(
        classification=nn.BCEWithLogitsLoss(),
        jigsaw=nn.CrossEntropyLoss(),
        #coloring=nn.MSELoss(),
        coloring=WeightedMSELoss(reduction='mean')
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    if cfg.freeze_modules['blocks']==True:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,  # or slightly higher if your head is already stable
            steps_per_epoch=len(train_dataloader),
            epochs=cfg.epochs,
            pct_start=0.1,  # % of total steps to ramp up LR
            anneal_strategy='cos',  # cosine decay
            div_factor=25.0,  # initial LR = max_lr / div_factor
            final_div_factor=1e4,  # end LR = max_lr / final_div_factor
        )

    #combine_losses = lambda x,y: x.sum()

    #print(f"Training with active heads: {' '.join(active_heads)}")
    print('='*100)
    logger.log(f"Training with active heads: {' '.join([head for head, v in cfg.active_heads.items() if v])}")
    logger.log(model.count_params_by_block())
    print('='*100)
    
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg.epochs,
        active_heads=[head for head, v in cfg.active_heads.items() if v],
        combine_losses=simple_combine_losses,
        accuracy_fun=multilabel_recall,
        logger=logger,
        threshold=0.5
    )



if __name__ == "__main__":
    main()

