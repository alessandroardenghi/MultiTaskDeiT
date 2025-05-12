import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import models.model_registry
import timm
import os
import numpy as np
from dataset_functions.classification import ClassificationDataset, MultiTaskDataset
from munch import Munch
from utils import AverageMeter, JigsawAccuracy
from models.full_model import MultiTaskDeiT
from utils import load_model
from multitask_training import train_model
from utils import hamming_acc, freeze_components, recolor_images, load_partial_checkpoint, HuberLoss
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
    
    model = create_model('MultiTaskDeiT_tiny', 
                         do_jigsaw = do_jigsaw, 
                         do_classification = do_classification, 
                         do_coloring = do_coloring, 
                         pixel_shuffle = cfg.pixel_shuffle,
                         verbose = cfg.verbose,
                         pretrained = cfg.pretrained_backbone) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    if cfg.pretrained_checkpoint:
        load_partial_checkpoint(model, cfg.pretrained_checkpoint, cfg.verbose)
    freeze_components(model, component_names=cfg.modules_to_freeze, freeze=True, verbose=cfg.verbose)
    freeze_components(model, component_names=cfg.modules_to_unfreeze, freeze=False, verbose=cfg.verbose)
    
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #train_dataset = ClassificationDataset('data', split='train', transform=transform)
    #val_dataset = ClassificationDataset('data', split='val', transform=transform)

    train_dataset = MultiTaskDataset('data', split='train', img_size = cfg.img_size, num_patches=14)
    val_dataset = MultiTaskDataset('data', split='val', img_size = cfg.img_size, num_patches=14)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True)
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False)
    criterion = Munch(
        classification=nn.BCEWithLogitsLoss(),
        jigsaw=nn.CrossEntropyLoss(),
        #coloring=nn.MSELoss()
        coloring=HuberLoss()
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    combine_losses = lambda x,y: x.sum()


    ###### testtttttt #######
    # # Load the checkpoint
    # checkpoint_path = 'logs/run_20250510_1632/checkpoints/best_model.pth'
    # checkpoint = torch.load(checkpoint_path, map_location='cuda')  # or 'cuda' if available

    # # Restore model, optimizer, and epoch
    # model.load_state_dict(checkpoint['model_state_dict'])
    # recolor_images(data_path='data', output_dir='coloring_test', split='val', model=model, n_images=16, shuffle=True)
    # return
    ###### testtttttt #######


    #print(f"Training with active heads: {' '.join(active_heads)}")
    logger.log(f"Training with active heads: {' '.join(active_heads)}")
    logger.log(f'\nModel Parameters: \n{model.count_params_by_block()}')
    #print(model.count_params_by_block())
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=cfg.epochs,
        active_heads=active_heads,
        combine_losses=combine_losses,
        accuracy_fun=hamming_acc,
        logger=logger,
        threshold=0.5,
        save_path='models_saved',
    )



if __name__ == "__main__":
    main()

