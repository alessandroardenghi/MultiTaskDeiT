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

from functions.loss import WeightedL1Loss, WeightedMSELoss
from engine.multitask_training import train_model
from functions.logger import TrainingLogger
from functions.utils import *

def main():
    
    cfg = load_config('configs/eval/config_class.yaml')        # cfg dict with all attributes inside
    
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

    parentdir = 'model_results'
    classdir = 'class_metrics'
    outdir = os.path.join(parentdir, classdir)
    output_file = os.path.join(outdir, f'{cfg.output_dir}.json')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    if cfg.mask is not None:
        mask = cfg.mask
        mask.sort()
    else:
        mask = None
    
    threshold = 0.5
    accuracy = AverageMeter()
    rec_perimage = AverageMeter()
    prec_perimage = AverageMeter()
    total_tp = total_fp = total_fn = None

    for batch_idx, (imgs, labels) in enumerate(test_dataloader):
        imgs, labels = move_to_device(imgs, device), move_to_device(labels, device)
        outputs = model(imgs)
        out_logits = outputs.pred_cls # out: (B, n_classes)


        class_outputs = (torch.sigmoid(out_logits) > threshold).int()
        acc, total = multilabel_accuracy(class_outputs, labels.label_classification, mask=mask)
        accuracy.update(acc, total)
        r, total = multilabel_recall(class_outputs, labels.label_classification, mask=mask)
        p, total = multilabel_precision(class_outputs, labels.label_classification, mask=mask)
        rec_perimage.update(r, total)
        prec_perimage.update(p, total)

        tp, fp, fn = update_perclass_metrics(class_outputs, labels.label_classification, mask=mask)
        if total_tp is None:
            total_tp = tp
            total_fp = fp
            total_fn = fn
        else:
            total_tp += tp
            total_fp += fp
            total_fn += fn

    f1_perimage = multilabel_f1(prec_perimage.avg, rec_perimage.avg)
    precision, recall, f1, avg_f1 = compute_perclass_f1(total_tp, total_fp, total_fn)
    # Build per-metric dicts
    if mask is not None:
        precision_dict = {cl: precision[i] for i, cl in enumerate(mask)}
        recall_dict = {cl: recall[i] for i, cl in enumerate(mask)}
        f1_dict = {cl: f1[i] for i, cl in enumerate(mask)}
    else:
        precision_dict = {i: precision[i] for i in range(len(precision))}
        recall_dict = {i: recall[i] for i in range(len(recall))}
        f1_dict = {i: f1[i] for i in range(len(f1))}
    # Sort each dict by value descending
    precision_sorted = dict(sorted(precision_dict.items(), key=lambda x: x[1], reverse=True))
    recall_sorted = dict(sorted(recall_dict.items(), key=lambda x: x[1], reverse=True))
    f1_sorted = dict(sorted(f1_dict.items(), key=lambda x: x[1], reverse=True))    
    

    print(f'Information for {cfg.experiment_name}')
    print(f"Accuracy: {accuracy.avg:.4f} ({accuracy.sum}/{len(test_dataset)})")
    print(f"Recall per Image: {rec_perimage.avg:.4f}")
    print(f"Precision per Image: {prec_perimage.avg:.4f}")
    print(f'F1 score per Image: {f1_perimage:.4f}')
    print(f'Per-Class F1 score: {f1_sorted}')

    # save statistics
    metrics = {
        "accuracy": accuracy.avg,
        "recall_per_image": rec_perimage.avg,
        "precision_per_image": prec_perimage.avg,
        "f1_per_image": f1_perimage,
        "precision_per_class": precision_sorted,
        "recall_per_class": recall_sorted,
        "f1_per_class": f1_sorted,
        "avg_f1": avg_f1,
    }

    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)



if __name__ == "__main__":
    main()

