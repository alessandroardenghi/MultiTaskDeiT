# information to log in training
# - create directory for the model
# - create directory for the model checkpoints for best validation accuracy (best model)
# - epoch
# - train_loss
# - val_loss
# - train_accuracy
# - val_accuracy
# - information about the model included in the yaml file
# - sum up of the accuracy and loss for each head
# - save the final model
# - save the best model

import os
import logging
import yaml
import json
import torch
from munch import Munch
from datetime import datetime


class TrainingLogger:
    """
    A class to handle logging and checkpointing during training.
    It creates directories for logs and checkpoints, sets up logging,
    and provides methods to log metrics, save configurations, and save checkpoints.
    Attributes:
        log_dir (str): Directory for logs.
        checkpoint_dir (str): Directory for checkpoints.
        logger (logging.Logger): Logger instance.
        current_best_loss (float): Current best validation loss.
        save_step (int): Number of epochs after which to save metrics.
        log_config (Munch): Configuration settings for the training run.
        log_metrics (list): List to store metrics for each epoch.
    Methods:
        log(message, level): Log a message at the specified level.
        log_epoch(epoch, train_metrics, val_metrics, save_log): Log metrics for the current epoch.
        best_model(model, optimizer, epoch, val_loss, name): Save the model if it has the best validation loss.
        save_config(config, filename): Save the training configuration to a YAML file.
        save_checkpoint(model, optimizer, epoch, name): Save the model and optimizer state.
        save_metrics(filename): Save metrics to a JSON file.
    """
    
    def __init__(self, save_step=15, base_log_dir="logs", experiment_name=None):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.experiment_name = experiment_name or f"run_{self.timestamp}"

        self.log_dir = os.path.join(base_log_dir, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._setup_logger()
        self.current_best_loss = float('inf')
        self.save_step = save_step
        self.log_config = {}
        self.log_metrics = []

    def _setup_logger(self):
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, message, level="info"):
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)

    def log_epoch(self, epoch: int, train_metrics: Munch, val_metrics: Munch, save_log=True):
        self.log_metrics.append({
            "epoch": epoch,
            "train_loss": train_metrics.train_epoch_loss,
            "train_classification_loss": train_metrics.train_epoch_classification_loss,
            "train_coloring_loss": train_metrics.train_epoch_coloring_loss,
            "train_jigsaw_loss": train_metrics.train_epoch_jigsaw_loss,
            "train_class_accuracy": train_metrics.train_epoch_class_accurary,
            "train_jigsaw_pos_accuracy": train_metrics.train_epoch_jigsaw_pos_accuracy,
            "train_jigsaw_pos_topnaccuracy": train_metrics.train_epoch_jigsaw_pos_topnaccuracy,
            "train_jigsaw_rot_accuracy": train_metrics.train_epoch_jigsaw_rot_accuracy,
            "val_loss": val_metrics.val_epoch_loss,
            "val_classification_loss": val_metrics.val_epoch_classification_loss,
            "val_coloring_loss": val_metrics.val_epoch_coloring_loss,
            "val_jigsaw_loss": val_metrics.val_epoch_jigsaw_loss,
            "val_class_accuracy": val_metrics.val_epoch_class_accurary,
            "val_jigsaw_pos_accuracy": val_metrics.val_epoch_jigsaw_pos_accuracy,
            "val_jigsaw_pos_topnaccuracy": val_metrics.val_epoch_jigsaw_pos_topnaccuracy,
            "val_jigsaw_rot_accuracy": val_metrics.val_epoch_jigsaw_rot_accuracy
        })

        if (len(self.log_config)+1 >= self.save_step) or save_log:
            self.save_metrics(filename="metrics.json")
            self.log_metrics = []

        self.log(
            f"\nEpoch {epoch}/{self.log_config.epochs}"
            + "\n"
            + f"Train Loss: {train_metrics.train_epoch_loss:.4f} | "
            + f"Train Classification Loss: {train_metrics.train_epoch_classification_loss:.4f} | "
            + f"Train Coloring Loss: {train_metrics.train_epoch_coloring_loss:.4f} | "
            + f"Train Jigsaw Loss: {train_metrics.train_epoch_jigsaw_loss:.4f}\n"
            + f"Train Class Acc: {train_metrics.train_epoch_class_accurary:.4f} | "
            + f"Train Jigsaw Pos Acc: {train_metrics.train_epoch_jigsaw_pos_accuracy:.4f} | "
            + f"Train Jigsaw Pos TopN Acc: {train_metrics.train_epoch_jigsaw_pos_topnaccuracy:.4f} | "
            + f"Train Jigsaw Rot Acc: {train_metrics.train_epoch_jigsaw_rot_accuracy:.4f}\n"
            + "=" * 50 + "\n"
            + f"Val Loss: {val_metrics.val_epoch_loss:.4f} | "
            + f"Val Classification Loss: {val_metrics.val_epoch_classification_loss:.4f} | "
            + f"Val Coloring Loss: {val_metrics.val_epoch_coloring_loss:.4f} | "
            + f"Val Jigsaw Loss: {val_metrics.val_epoch_jigsaw_loss:.4f}\n"
            + f"Val Class Acc: {val_metrics.val_epoch_class_accurary:.4f} | "
            + f"Val Jigsaw Pos Acc: {val_metrics.val_epoch_jigsaw_pos_accuracy:.4f} | "
            + f"Val Jigsaw Pos TopN Acc: {val_metrics.val_epoch_jigsaw_pos_topnaccuracy:.4f} | "
            + f"Val Jigsaw Rot Acc: {val_metrics.val_epoch_jigsaw_rot_accuracy:.4f}",
            level="info"
        )

    def best_model(self, model, optimizer, epoch, val_loss, name="best_model.pth"):
        if self.current_best_loss > val_loss:
            self.save_checkpoint(model, optimizer, epoch, name)

    def save_config(self, config: Munch, filename="config.yaml"):
        self.log_config = config

        config_path = os.path.join(self.log_dir, filename)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self.log(f"Saved configuration to {config_path}")

    def save_checkpoint(self, model, optimizer, epoch, name="checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, name)
        torch.save(checkpoint, checkpoint_path)
        if name != "best_model.pth":
            self.log(f"Saved checkpoint to {checkpoint_path}")

    def save_metrics(self, filename="metrics.json"):
        path = os.path.join(self.log_dir, filename)

        # If the file exists, load existing metrics
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        else:
            existing = []

        # Combine old and new metrics
        combined = existing + self.log_metrics

        # Write updated metrics back to file
        with open(path, "w") as f:
            json.dump(combined, f, indent=4)
        
        self.log(f"Saved metrics to {path}")