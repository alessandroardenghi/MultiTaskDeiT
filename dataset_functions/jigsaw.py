import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from dataclasses import dataclass
from utils import jigsaw_image


# REMOVING FLIPPING ?


@dataclass
class JigsawConfig:
    n_patches: int = 2
    jigsaw: bool = False
    rotation: bool = False
    flip: bool = False
    noise_mean: int = 0
    noise_std: int = 0

class JigsawDataset(Dataset):
    
    def __init__(self, 
                 data_dir: str, 
                 config: JigsawConfig,
                 split: str ='train', 
                 transform=None):
        
        if split not in ['train', 'val', 'test']:
            raise Exception('Split must be chosen between train, test and val.')
        
        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        
        self.config = config
        self.noise_params = (config.noise_mean, config.noise_std)
        self.image_dir = os.path.join(data_dir, 'images')
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        pil_image = Image.open(image_path).convert('RGB')
        if self.transform:
            tensor_image = self.transform(pil_image)
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor_image)    
        
        transformation_vec, jigsaw_img = jigsaw_image(np.array(pil_image.convert('L')), 
                                                      n = self.config.n_patches, 
                                                      jigsaw=self.config.jigsaw, 
                                                      rotation=self.config.rotation, 
                                                      noise_mean= self.config.noise_mean,
                                                      noise_std=self.config.noise_std)
        jigsaw_img = torch.from_numpy(jigsaw_img).unsqueeze(0)
        
        return jigsaw_img, tensor_image, torch.tensor(transformation_vec)
        
        