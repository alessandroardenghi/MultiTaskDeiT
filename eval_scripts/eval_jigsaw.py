import os
import json
import numpy as np
from munch import Munch
import timm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from engine.multitask_dataloader import MultiTaskDataset

from timm import create_model
import models.model_registry
from functions.utils import *

def main():
    
    cfg = load_config('configs/eval/config_jigsaw.yaml')        # cfg dict with all attributes inside
    
    model = create_model(cfg.model_name, 
                        n_classes = cfg.classification_cfg.n_classes,
                         img_size = cfg.img_size,
                         do_jigsaw = cfg.active_heads.jigsaw, 
                         pretrained = True,                # TO BE MODIFIED WITH TRUE!
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
                                transform=True)

    test_dataloader = DataLoader(test_dataset, 
                                batch_size=cfg.batch_size, 
                                shuffle=False,
                                num_workers=cfg.n_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()


    
    parentdir = 'model_results'
    classdir = 'jigsaw_metrics'
    outdir = os.path.join(parentdir, classdir)
    output_file = os.path.join(outdir, f'{cfg.output_dir}.json')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
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

