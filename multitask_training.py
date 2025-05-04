import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
import os
import numpy as np
from dataset_functions.classification import ClassificationDataset


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: dict, 
    optimizer: torch.optim.Optimizer,
    device: torch.device, 
    epoch: int,
    active_heads: list # control which heads are active
    ):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (Iterable): The data loader for the training data.
        criterion (dict): The loss functions for each head: {'head_name': loss_function}.
                            1. 'classification': nn.BCEWithLogitsLoss, LabelSmoothingCrossEntropy
                            2. 'coloring': nn.MSELoss
                            3. 'jigsaw': nn.MSELoss if using a regression head otherise nn.CrossEntropyLoss 
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training.
        epoch (int): The current epoch number.
        active_heads (Optional[list]): List of active heads to train. If None, all heads are trained.
    Returns:
        epoch_loss (float): The average loss for the epoch.
        epoch_acc (float): The average accuracy for the epoch.
    """

    model.train()

    assert all(head in criterion.keys() for head in active_heads)

    loss_m = AverageMeter()  # For tracking the overall loss
    loss_m_classification = AverageMeter()  # For tracking the classification loss
    loss_m_coloring = AverageMeter()  # For tracking the coloring loss
    loss_m_jigsaw = AverageMeter()  # For tracking the jigsaw loss

    acc_m_classification = AverageMeter() # For tracking the classification accuracy
    acc_m_jigsaw = AverageMeter() # For tracking the jigsaw accuracy

    for batch_index, (images, labels) in enumerate(tqdm(data_loader, desc="Training", leave=False)):

        images, labels = images.to(device), labels.to(device).float()

        





def validate():
    pass

