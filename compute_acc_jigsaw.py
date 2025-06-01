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
from utils import multilabel_accuracy, multilabel_recall, multilabel_precision, multilabel_f1
from utils import compute_perclass_f1, update_perclass_metrics, AverageMeter, JigsawAccuracy
import json
from utils import freeze_components
from timm import create_model
from logger import TrainingLogger

#from torch.optim.lr_scheduler import OneCycleLR
def move_to_device(munch_obj, device):
    return Munch({k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in munch_obj.items()})

import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Munch.fromDict(cfg)

def main():
    
    cfg = load_config('config_class.yaml')        # cfg dict with all attributes inside
    
    if cfg.weights != '':
        weights = torch.from_numpy(np.load(cfg.weights))
    else:
        weights = None
    
    model = create_model(cfg.model_name, 
                        n_classes = cfg.classification_cfg.n_classes,
                         img_size = cfg.img_size,
                         do_jigsaw = cfg.active_heads.jigsaw, 
                         pretrained = True,
                         do_classification = cfg.active_heads.classification, 
                         do_coloring = cfg.active_heads.coloring, 
                         jigsaw_cfg = cfg.jigsaw_cfg,
                         pixel_shuffle_cfg = cfg.pixel_shuffle_cfg,
                         verbose = cfg.verbose,
                         pretrained_model_info = cfg.pretrained_info) # /home/3141445/.cache/torch/hub/checkpoints/deit_tiny_patch16_224-a1311bcf.pth
    
    
    freeze_components(model, component_names=[module for module, v in cfg.freeze_modules.items() if v], freeze=True, verbose=cfg.verbose)
    freeze_components(model, component_names=[module for module, v in cfg.unfreeze_modules.items() if v], freeze=False, verbose=cfg.verbose)
    
    test_dataset = MultiTaskDataset(cfg.data_path, 
                                split='test', 
                                img_size = cfg.img_size, 
                                num_patches=cfg.jigsaw_cfg.n_jigsaw_patches, 
                                do_rotate=True,
                                do_jigsaw=cfg.active_heads.jigsaw,
                                do_coloring=cfg.active_heads.coloring,
                                do_classification=cfg.active_heads.classification,
                                weights=weights, 
                                transform=True)

    test_dataloader = DataLoader(test_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False,
                                num_workers=cfg.n_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()


    output_file = os.path.join('jigsaw_metrics', f'{cfg.experiment_name}.json')
    if not os.path.exists('jigsaw_metrics'):
        os.makedirs('jigsaw_metrics')
    
    acc_m_pos = JigsawAccuracy(n=3) # For tracking the jigsaw accuracy in predicting positions
    acc_m_rot = JigsawAccuracy(n=1) # For tracking the jigsaw accuracy in predicting rotations

    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        imgs, labels = move_to_device(imgs, device), move_to_device(labels, device)
        outputs = model(imgs)
        out_logits = outputs.pred_jigsaw_pos

        acc_m_pos.update(outputs.pred_jigsaw_pos, labels.pos_vec)
        acc_m_rot.update(outputs.pred_jigsaw_rot, labels.rot_vec)

    # print statitics
    jigsaw_pos_accuracy = acc_m_pos.get_scores()['accuracy']
    try:
        jigsaw_pos_topnaccuracy = acc_m_pos.get_scores()['topn_accuracy']
    except:
        jigsaw_pos_topnaccuracy = 0
    jigsaw_rot_accuracy = acc_m_rot.get_scores()['accuracy']

    print(f'Jigsaw position accuracy: {jigsaw_pos_accuracy:.4f}')
    print(f'Jigsaw position top3 accuracy: {jigsaw_pos_topnaccuracy:.4f}')
    print(f'Jigsaw rotation accuracy: {jigsaw_rot_accuracy:.4f}')

    # save statistics
    metrics = {
        'jigsaw_pos_accuracy': jigsaw_pos_accuracy,
        'jigsaw_pos_top3accuracy': jigsaw_pos_topnaccuracy,
        'jigsaw_rot_accuracy': jigsaw_rot_accuracy
    }
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)

   
if __name__ == "__main__":
    main()

