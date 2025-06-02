import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import models.model_registry
import os
import numpy as np
from multitask_dataloader import MultiTaskDataset
from munch import Munch
from timm import create_model
from PIL import Image
import cv2
import numpy as np
import yaml
from munch import Munch

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Munch.fromDict(cfg)


def recolor_image(L, ab):
    L_denorm = L[0].numpy() * 255.0  
    ab_denorm = (ab.numpy() * 127.0) + 128   
    
    L_denorm = np.clip(L_denorm, 0, 255)
    ab_denorm = np.clip(ab_denorm, 0, 255)

    ab_denorm = np.transpose(ab_denorm, (1, 2, 0))  
    lab_denorm = np.concatenate([L_denorm[..., np.newaxis], ab_denorm], axis=-1) 
    
    rgb_reconstructed = cv2.cvtColor(lab_denorm.astype(np.uint8), cv2.COLOR_LAB2RGB)
    rgb_reconstructed = np.clip(rgb_reconstructed, 0, 255).astype(np.uint8)

    return rgb_reconstructed

def recolor_images(data_path, output_dir, split, model, n_images, img_size, shuffle=False):
    os.makedirs(output_dir, exist_ok=True)
    dataset = MultiTaskDataset(data_path, split=split, img_size=img_size, do_coloring=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    for i, (images, labels) in enumerate(loader):
        print(images.keys())
        if i >= n_images:
            break
        output = model(images)

        # Assuming images.original_image[0] is a torch.Tensor
        colored_img = recolor_image(images.image_colorization[0].detach(), output.pred_coloring[0].detach())
        colored_img = Image.fromarray(colored_img)
        colored_img.save(os.path.join(output_dir,f'image{i}.jpg'))


def main():
    
    cfg = load_config('configs/eval/config_coloring.yaml')        # cfg dict with all attributes inside
    
    model = create_model(cfg.model_name, 
                         img_size = cfg.img_size,
                         do_jigsaw = False, 
                         pretrained = False,            # TO BE SUBSTITUTED WITH TRUE!
                         n_classes = 80,
                         do_classification = False, 
                         do_coloring = True, 
                         jigsaw_cfg = cfg.jigsaw_cfg,
                         pixel_shuffle_cfg = cfg.pixel_shuffle_cfg,
                         verbose = cfg.verbose,
                         pretrained_model_info = cfg.pretrained_info)
    output_dir = os.path.join('coloring_results', cfg.output_dir)
    recolor_images(data_path=cfg.data_path, 
                   output_dir=output_dir, 
                   split='test', model=model, 
                   n_images=cfg.n_images, 
                   shuffle=cfg.shuffle, 
                   img_size=cfg.img_size)
    return

if __name__ == '__main__':
    main()