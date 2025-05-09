import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import models.model_registry
import timm
import os
import numpy as np
from dataset_functions.classification import ClassificationDataset
from munch import Munch
from utils import AverageMeter, JigsawAccuracy
from models.full_model import MultiTaskDeiT
from utils import load_model
from multitask_training import train_model
from utils import hamming_acc
from timm import create_model

def main():

    #active_heads = ['classification', 'coloring', 'jigsaw'] 
    active_heads = ['coloring']

    model = create_model('MultiTaskDeiT_tiny', 
                         do_jigsaw=False, 
                         do_classification=False, 
                         do_coloring=True, 
                         pixel_shuffle=False,
                         verbose=False,
                         pretrained=True) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ClassificationDataset('data', split='train', transform=transform)
    val_dataset = ClassificationDataset('data', split='val', transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    criterion = Munch(
        classification=nn.BCEWithLogitsLoss(),
        jigsaw=nn.CrossEntropyLoss(),
        coloring=nn.MSELoss()
    )
    num_epochs = 30
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    combine_losses = lambda x,y: x.sum()
    save_path = 'temp_checkpoints/coloring'
    

    ###### testtttttt #######
    def print_weighted_blocks(model):
        for name, module in model.named_modules():
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                print(name)
    
    #print(print_weighted_blocks(model))

    #return
    ###### testtttttt #######


    print(f"Training with active heads: {' '.join(active_heads)}")
    print(model.count_params_by_block())
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        active_heads=active_heads,
        combine_losses=combine_losses,
        accuracy_fun=hamming_acc,
        threshold=0.5,
        save_path=save_path,
    )



if __name__ == "__main__":
    main()

