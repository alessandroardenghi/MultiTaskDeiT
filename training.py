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



class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # The current value
        self.avg = 0  # The running average
        self.sum = 0  # The total sum of values
        self.count = 0  # The number of values processed

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # Sum of values
        self.count += n  # Count of samples processed
        self.avg = self.sum / self.count  # Running average

def hamming_loss(y_true, y_pred):
    """
    Compute the Hamming loss for multi-label classification.

    Parameters:
    - y_true (Tensor): Ground truth labels of shape [batch_size, num_labels].
    - y_pred (Tensor): Predicted probabilities (after sigmoid) of shape [batch_size, num_labels].
    - threshold (float): The threshold to convert probabilities to binary predictions (default 0.5).

    Returns:
    - hamming_loss (float): The Hamming loss for the batch.
    """

    # Calculate the number of incorrect labels
    incorrect_labels = (y_pred != y_true).float()

    # Calculate Hamming loss (average over all labels and samples)
    hamming_loss_value = incorrect_labels.sum() / (y_true.numel())

    return hamming_loss_value

def train_one_epoch(
    model, 
    loader, 
    criterion, 
    optimizer, 
    accuracy_fun, 
    device):


    model.train()
    threshold = 0.5
    loss_m = AverageMeter()  # For tracking the loss
    acc_m = AverageMeter() # For tracking the accuracy

    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device), labels.to(device).float()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        loss_m.update(loss.item(), images.shape[0])
        sigmoid_outputs = torch.sigmoid(outputs)
        preds = (sigmoid_outputs > threshold).int()
        acc_m.update(accuracy_fun(preds, labels), images.shape[0])

    epoch_loss = loss_m.avg
    epoch_acc = acc_m.avg
    return epoch_loss, epoch_acc

def validate(
    model, 
    loader, 
    criterion, 
    accuracy_fun, 
    device):

    model.eval()
    threshold = 0.5
    loss_m = AverageMeter()  # For tracking the loss
    acc_m = AverageMeter() # For tracking the accuracy

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Validation", leave=False)):
          images, labels = images.to(device), labels.to(device).float()

          outputs = model(images)
          loss = criterion(outputs, labels)

          loss_m.update(loss.item(), images.shape[0])
          sigmoid_outputs = torch.sigmoid(outputs)
          preds = (sigmoid_outputs > threshold).int()
          acc_m.update(accuracy_fun(preds, labels), images.shape[0])

    return loss_m.avg, acc_m.avg

def train_model(
    model, 
    train_dataloader, 
    val_dataloader, 
    criterion, 
    optimizer, 
    accuracy_fun, 
    num_epochs,
    path=None,
    device=None):
    """
    Train the model for a specified number of epochs.
    Parameters:
    - model (nn.Module): The model to be trained.
    - train_dataloader (DataLoader): DataLoader for the training dataset.
    - val_dataloader (DataLoader): DataLoader for the validation dataset.
    - criterion (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - accuracy_fun (function): Function to compute accuracy.
    - num_epochs (int): Number of epochs to train the model.
    - device (torch.device, optional): Device to run the model on. If None, will use GPU if available.
    Returns:
    - None
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(model=model,
                                                loader=train_dataloader,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                accuracy_fun=accuracy_fun,
                                                device=device)
        val_loss, val_acc = validate(model=model,
                                    loader=val_dataloader,
                                    criterion=torch.nn.BCEWithLogitsLoss(),
                                    accuracy_fun=hamming_loss,
                                    device=device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    
    save_model(model, path=path)

    

def save_model(model, path=None):
    """
    Save the model to a specified path.
    Parameters:
    - model (nn.Module): The model to be saved.
    - path (str): Path to save the model.
    Returns:
    - None
    """
    
    name = model.__class__.__name__
    if path is None:
        path = f"{name}.pth"
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + f"/{name}.pth"

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_arch, model_name, path=None):
    """
    Load the model from a specified path.
    Parameters:
    - model_arch (nn.Module): The architecture of the model.
    - model (str): The model to be loaded.
    - path (str): Path to load the model from.
    Returns:
    - model (nn.Module): The loaded model.
    """
    
    if path is None:
        path = f"{model_name}.pth"
    else:
        path = path + f"/{model_name}.pth"
    
    model.load_state_dict(torch.load(path, weight_only=True))
    print(f"Model loaded from {path}")
    return model
    

def main():

    transform = transforms.Compose(
    [
        transforms.Resize([224,224]),
        transforms.ToTensor(),
    ]
    )

    train_dataset = ClassificationDataset('data', split='train', transform=transform)
    val_dataset = ClassificationDataset('data', split='val', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    name = 'deit_tiny_patch16_224.fb_in1k'
    model = timm.create_model(name, pretrained=True)
    num_features = model.head.in_features  
    custom_head = torch.nn.Sequential(
        torch.nn.Linear(num_features, 512),  
        torch.nn.ReLU(),
        torch.nn.Linear(512, 20)  
    )
    model.head = custom_head
    for name, param in model.named_parameters():
        if 'head' not in name:  
            param.requires_grad = False

    criterion = nn.BCE()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                accuracy_fun=hamming_loss,
                num_epochs=5,
                path='./benchmark_models',)


if __name__ == "__main__":
    main()