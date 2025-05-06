import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
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

def main():

    active_heads = ['classification', 'coloring', 'jigsaw'] 

    model = MultiTaskDeiT(
        do_jigsaw=True,
        do_coloring=True,
        do_classification=True,
        n_classes=20,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=86,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=None
    )
    #print(model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    num_epochs = 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    combine_losses = lambda x,y: x.sum()
    save_path = None

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
        save_path=None,
    )



if __name__ == "__main__":
    main()

