import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections.abc import Iterable, Callable
from typing import List
from torchvision import datasets, transforms
from tqdm import tqdm
import timm
import os
import numpy as np
from dataset_functions.classification import ClassificationDataset
from munch import Munch
from utils import AverageMeter, JigsawAccuracy, save_model

def move_to_device(munch_obj, device):
    return Munch({k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in munch_obj.items()})

def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: Munch,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    active_heads: List[str],
    combine_losses: Callable,
    accuracy_fun: Callable,
    threshold: float = 0.5,
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
        combine_losses (callable): Function to combine the losses from different heads.
        accuracy_fun (callable): Function to calculate classification accuracy.
        threshold (float): Threshold for classification accuracy.
    Returns:
        epoch_loss (float): The average loss for the epoch.
        epoch_classification_loss (float): The average classification loss for the epoch.
        epoch_coloring_loss (float): The average coloring loss for the epoch.
        epoch_jigsaw_loss (float): The average jigsaw loss for the epoch.
        epoch_class_accurary (float): The average classification accuracy for the epoch.
        epoch_jigsaw_pos_accuracy (float): The average jigsaw position accuracy for the epoch.
        epoch_jigsaw_rot_accuracy (float): The average jigsaw rotation accuracy for the epoch.
    """

    assert all(head in criterion.keys() for head in active_heads)

    model.train()
    #model = model.to(device)

    loss_m = AverageMeter()  # For tracking the overall loss
    loss_m_classification = AverageMeter()  # For tracking the classification loss
    loss_m_coloring = AverageMeter()  # For tracking the coloring loss
    loss_m_jigsaw = AverageMeter()  # For tracking the jigsaw loss

    acc_m_classification = AverageMeter() # For tracking the classification accuracy
    acc_m_pos = JigsawAccuracy(n=5) # For tracking the jigsaw accuracy in predicting positions
    acc_m_rot = JigsawAccuracy(n=1) # For tracking the jigsaw accuracy in predicting rotations

    for batch_index, (images, labels) in enumerate(tqdm(data_loader, desc="Training", leave=False)):
        
        images = move_to_device(images, device)
        labels = move_to_device(labels, device)
        
        
        # Forward and losses calculation
        outputs = model(images)
        # print(f'OUTPUTS PRED CLSDEVICE: {outputs.pred_cls.device}')
        # print(f'OUTPUTS PRED COLORING DEVICE: {outputs.pred_coloring.device}')
        # print(f'OUTPUTS PRED JIGSAW: {outputs.pred_jigsaw.device}')
        # print(f'OUTPUTS POS VEC JIGSAW: {outputs.pos_vector.device}')
        # print(f'OUTPUTS ROT VEC JIGSAW: {outputs.rot_vector.device}')
        
        losses = torch.zeros(3).to(device)
        
        if 'classification' in active_heads:
            class_loss = criterion.classification(outputs.pred_cls, labels.label_classification)
            loss_m_classification.update(class_loss.item(), images.image_classification.shape[0])
            losses[0] = class_loss
        
        if 'coloring' in active_heads:
            coloring_loss = criterion.coloring(outputs.pred_coloring, labels.ab_channels)
            loss_m_coloring.update(coloring_loss.item(), images.image_colorization.shape[0])
            losses[1] = coloring_loss
        
        if 'jigsaw' in active_heads:
            B,P,C = outputs.pred_jigsaw.shape
            H = W = int(P**0.5)
            pred_jigsaw2d = outputs.pred_jigsaw.view(B,H,W,C).permute(0, 3, 1, 2)
            pos_vector = labels.pos_vec.view(B,H,W)
            rot_vector = labels.rot_vec.view(B,H,W)
              
            jigsaw_loss = 0.5 * criterion.jigsaw(pred_jigsaw2d[:,:P,:,:], pos_vector) + \
                            0.5 * criterion.jigsaw(pred_jigsaw2d[:,P:,:,:], rot_vector)
            loss_m_jigsaw.update(jigsaw_loss.item(), images.image_jigsaw.shape[0])
            losses[2] = jigsaw_loss

        # Combine losses
        loss = combine_losses(losses, active_heads) ##TODO: write combine_losses function
        loss_m.update(loss.item(), images.image_classification.shape[0])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        if 'classification' in active_heads:
            class_outputs = (torch.sigmoid(outputs.pred_cls) > threshold).int()
            acc_m_classification.update(accuracy_fun(class_outputs, labels.label_classification), images.image_classification.shape[0])  
        if 'jigsaw' in active_heads:
            acc_m_pos.update(outputs.pred_jigsaw[:,:,:P], labels.pos_vec)
            acc_m_rot.update(outputs.pred_jigsaw[:,:,P:], labels.rot_vec)
    
    epoch_loss = loss_m.avg
    epoch_classification_loss = loss_m_classification.avg
    epoch_coloring_loss = loss_m_coloring.avg
    epoch_jigsaw_loss = loss_m_jigsaw.avg
    epoch_class_accurary = acc_m_classification.avg
    epoch_jigsaw_pos_accuracy = acc_m_pos.get_scores()['accuracy']
    try:
        epoch_jigsaw_pos_topnaccuracy = acc_m_pos.get_scores()['topn_accuracy']
    except:
        epoch_jigsaw_pos_topnaccuracy = 0
    epoch_jigsaw_rot_accuracy = acc_m_rot.get_scores()['accuracy']

    return Munch(
            train_epoch_loss=epoch_loss,
            train_epoch_classification_loss=epoch_classification_loss,
            train_epoch_coloring_loss=epoch_coloring_loss,
            train_epoch_jigsaw_loss=epoch_jigsaw_loss,
            train_epoch_class_accurary=epoch_class_accurary,
            train_epoch_jigsaw_pos_accuracy=epoch_jigsaw_pos_accuracy,
            train_epoch_jigsaw_pos_topnaccuracy=epoch_jigsaw_pos_topnaccuracy,
            train_epoch_jigsaw_rot_accuracy=epoch_jigsaw_rot_accuracy
        )


def validate(
    model: torch.nn.Module,
    data_loader: Iterable, 
    criterion: Munch,
    device: torch.device, 
    active_heads: list, # control which heads are active
    accuracy_fun: callable, # function to calculate classification accuracy
    combine_losses: callable,
    threshold: float = 0.5, # threshold for classification,
    ):
    """
    Validate the model on the validation set.
    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (Iterable): The data loader for the validation data.
        criterion (dict): The loss functions for each head: {'head_name': loss_function}.
                            1. 'classification': nn.BCEWithLogitsLoss, LabelSmoothingCrossEntropy
                            2. 'coloring': nn.MSELoss
                            3. 'jigsaw': nn.MSELoss if using a regression head otherise nn.CrossEntropyLoss 
        device (torch.device): The device to use for validation.
        active_heads (Optional[list]): List of active heads to validate. If None, all heads are validated.
        accuracy_fun (callable): Function to calculate classification accuracy.
        combine_losses (callable): Function to combine the losses from different heads.
        threshold (float): Threshold for classification accuracy.
    Returns:
    """

    assert all(head in criterion.keys() for head in active_heads)

    model.eval()
    #model = model.to(device)
    
    loss_m = AverageMeter()  # For tracking the overall loss
    loss_m_classification = AverageMeter()  # For tracking the classification loss
    loss_m_coloring = AverageMeter()  # For tracking the coloring loss
    loss_m_jigsaw = AverageMeter()  # For tracking the jigsaw loss

    acc_m_classification = AverageMeter() # For tracking the classification accuracy
    acc_m_pos = JigsawAccuracy(n=5) # For tracking the jigsaw accuracy in predicting positions
    acc_m_rot = JigsawAccuracy(n=1) # For tracking the jigsaw accuracy in predicting rotations

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Validation", leave=False)):
            
            images = move_to_device(images, device)
            labels = move_to_device(labels, device)

            # Forward and losses calculation
            outputs = model(images)

            losses = torch.zeros(3).to(device)
            if 'classification' in active_heads:
                class_loss = criterion.classification(outputs.pred_cls, labels.label_classification)
                loss_m_classification.update(class_loss.item(), images.image_classification.shape[0])
                losses[0] = class_loss
            
            if 'coloring' in active_heads:
                coloring_loss = criterion.coloring(outputs.pred_coloring, labels.ab_channels)
                loss_m_coloring.update(coloring_loss.item(), images.image_colorization.shape[0])
                losses[1] = coloring_loss
            
            if 'jigsaw' in active_heads:
                B,P,C = outputs.pred_jigsaw.shape
                H = W = int(P**0.5)
                pred_jigsaw2d = outputs.pred_jigsaw.view(B,H,W,C).permute(0, 3, 1, 2)
                pos_vector = labels.pos_vec.view(B,H,W)
                rot_vector = labels.rot_vec.view(B,H,W)
                
                jigsaw_loss = 0.5 * criterion.jigsaw(pred_jigsaw2d[:,:P,:,:], pos_vector) + \
                                0.5 * criterion.jigsaw(pred_jigsaw2d[:,P:,:,:], rot_vector)
                loss_m_jigsaw.update(jigsaw_loss.item(), images.image_jigsaw.shape[0])
                losses[2] = jigsaw_loss

            # Combine losses
            loss = combine_losses(losses, active_heads)
            loss_m.update(loss.item(), images.image_classification.shape[0])

            # Metrics
            if 'classification' in active_heads:
                class_outputs = (torch.sigmoid(outputs.pred_cls) > threshold).int()
                acc_m_classification.update(accuracy_fun(class_outputs, labels), images.shape[0])
            if 'jigsaw' in active_heads:
                acc_m_pos.update(outputs.pred_jigsaw[:,:,:P], pos_vector)
                acc_m_rot.update(outputs.pred_jigsaw[:,:,P:], rot_vector)
    
    epoch_loss = loss_m.avg
    epoch_classification_loss = loss_m_classification.avg
    epoch_coloring_loss = loss_m_coloring.avg
    epoch_jigsaw_loss = loss_m_jigsaw.avg
    epoch_class_accurary = acc_m_classification.avg
    epoch_jigsaw_pos_accuracy = acc_m_pos.get_scores()['accuracy']
    try:
        epoch_jigsaw_pos_topnaccuracy = acc_m_pos.get_scores()['topn_accuracy']
    except:
        epoch_jigsaw_pos_topnaccuracy = 0
    epoch_jigsaw_rot_accuracy = acc_m_rot.get_scores()['accuracy']
    
    return Munch(
            val_epoch_loss=epoch_loss,
            val_epoch_classification_loss=epoch_classification_loss,
            val_epoch_coloring_loss=epoch_coloring_loss,
            val_epoch_jigsaw_loss=epoch_jigsaw_loss,
            val_epoch_class_accurary=epoch_class_accurary,
            val_epoch_jigsaw_pos_accuracy=epoch_jigsaw_pos_accuracy,
            val_epoch_jigsaw_pos_topnaccuracy=epoch_jigsaw_pos_topnaccuracy,
            val_epoch_jigsaw_rot_accuracy=epoch_jigsaw_rot_accuracy
        )


def train_model(
    model: torch.nn.Module,
    train_dataloader: Iterable,
    val_dataloader: Iterable,
    criterion: Munch,
    optimizer: torch.optim.Optimizer,
    #device: torch.device,
    num_epochs: int,
    active_heads: list, # control which heads are active
    combine_losses: callable, # function to combine the losses (it is a partial function)
    accuracy_fun: callable, # function to calculate classification accuracy
    threshold: float = 0.5, # threshold for classification
    save_path: str = None, # path to save the model
    ):
    """
    Train the model for a specified number of epochs.
    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (Iterable): DataLoader for the training dataset.
        val_dataloader (Iterable): DataLoader for the validation dataset.
        criterion (dict): The loss functions for each head: {'head_name': loss_function}.
                            1. 'classification': nn.BCEWithLogitsLoss, LabelSmoothingCrossEntropy
                            2. 'coloring': nn.MSELoss
                            3. 'jigsaw': nn.MSELoss if using a regression head otherise nn.CrossEntropyLoss 
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): The device to use for training.
        num_epochs (int): Number of epochs to train the model.
        active_heads (Optional[list]): List of active heads to train. If None, all heads are trained.
        combine_losses (callable): Function to combine the losses from different heads.
        accuracy_fun (callable): Function to calculate classification accuracy.
        threshold (float): Threshold for classification accuracy.
        save_path (str): Path to save the model. If None, the model will not be saved.
    Returns:
        None
    """

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nStart training")
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            active_heads=active_heads,
            combine_losses=combine_losses,
            accuracy_fun=accuracy_fun,
            threshold=threshold
        )
        val_metrics = validate(
            model=model,
            data_loader=val_dataloader,
            criterion=criterion,
            device=device,
            active_heads=active_heads,
            accuracy_fun=accuracy_fun,
            combine_losses=combine_losses,
            threshold=threshold
        )

        # Print metrics
        print(f"Train Loss: {train_metrics.train_epoch_loss:.4f} | \
            Train Classification Loss: {train_metrics.train_epoch_classification_loss:.4f} | \
            Train Coloring Loss: {train_metrics.train_epoch_coloring_loss:.4f} | \
            Train Jigsaw Loss: {train_metrics.train_epoch_jigsaw_loss:.4f}")
        print(f"Train Class Acc: {train_metrics.train_epoch_class_accurary:.4f} | \
            Train Jigsaw Pos Acc: {train_metrics.train_epoch_jigsaw_pos_accuracy:.4f} | \
            Train Jigsaw Pos TopN Acc: {train_metrics.train_epoch_jigsaw_pos_topnaccuracy:.4f} | \
            Train Jigsaw Rot Acc: {train_metrics.train_epoch_jigsaw_rot_accuracy:.4f}")
        print('='*50)
        print(f"Val Loss: {val_metrics.val_epoch_loss:.4f} | \
            Val Classification Loss: {val_metrics.val_epoch_classification_loss:.4f} | \
            Val Coloring Loss: {val_metrics.val_epoch_coloring_loss:.4f} | \
            Val Jigsaw Loss: {val_metrics.val_epoch_jigsaw_loss:.4f}")
        print(f"Val Class Acc: {val_metrics.val_epoch_class_accurary:.4f} | \
            Val Jigsaw Pos Acc: {val_metrics.val_epoch_jigsaw_pos_accuracy:.4f} | \
            Val Jigsaw Pos TopN Acc: {val_metrics.val_epoch_jigsaw_pos_topnaccuracy:.4f} | \
            Val Jigsaw Rot Acc: {val_metrics.val_epoch_jigsaw_rot_accuracy:.4f}\n")
        print('='*170)

    print(f"Training completed")


    if save_path is not None:
        save_model(model, path=save_path)
