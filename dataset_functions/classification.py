import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from utils import jigsaw_single_image
from munch import Munch

def preprocess_for_coloring(pil_img):
    # Convert to LAB and split channels
    lab = pil_img.convert('LAB')
    lab_np = np.array(lab).astype(np.float32)

    L = lab_np[:, :, 0] / 255.0  # Normalize L to [0,1]
    ab = lab_np[:, :, 1:] - 128  # Center a/b to [-128, 127] → now [-128, 127]

    # Convert to tensors
    L_tensor = torch.from_numpy(L).unsqueeze(0)  # [1, H, W]
    ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)) / 128.0  # [2, H, W], normalized to [-1, 1]

    # Repeat L for 3-channel input if needed
    L_3ch_tensor = L_tensor.repeat(3, 1, 1)  # [3, H, W]

    return L_3ch_tensor, ab_tensor


class ClassificationDataset(Dataset):
    
    def __init__(self, data_dir, split='train', transform=None):
        if split not in ['train', 'val', 'test']:
            raise Exception('Split must be chosen between train, test and val.')
        
        self.image_dir = os.path.join(data_dir, 'images')
        npz_data = np.load(os.path.join(data_dir, 'labels.npz'), allow_pickle=True)
        self.labels_dict = npz_data['labels'].item()

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if image_name not in self.labels_dict.keys():
            print('ERROR')
        label = self.labels_dict[image_name]
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

class MultiTaskDataset(Dataset):
    
    def __init__(self, data_dir, img_size = 224, split='train', transform_classification=None, num_patches = 14):
        if split not in ['train', 'val', 'test']:
            raise Exception('Split must be chosen between train, test and val.')
        
        self.img_size = img_size
        self.image_dir = os.path.join(data_dir, 'images')
        npz_data = np.load(os.path.join(data_dir, 'labels.npz'), allow_pickle=True)
        self.labels_dict = npz_data['labels'].item()
        self.num_patches = num_patches

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
        self.transform_classification = transform_classification if transform_classification is not None else transforms.ToTensor()
        #self.transform_coloring = transform_coloring if transform_coloring is not None else transforms.ToTensor()
        #self.transform_jigsaw = transform_jigsaw if transform_jigsaw is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        
        #image_classification = self.transform_classification(image)
    
        image_coloring, ab_channels = preprocess_for_coloring(image)
        image_classification = image_coloring.clone()
        image_jigsaw, pos_vec, rot_vec = jigsaw_single_image(image_coloring, n_patches = self.num_patches)
    
        
        if image_name not in self.labels_dict.keys():
            raise Exception('LABEL NOT FOUND')
            
        label_classification = self.labels_dict[image_name]
        label_classification = torch.tensor(label_classification, dtype=torch.float32)
        
        images = Munch(image_classification=image_classification,
                        image_colorization=image_coloring,
                        image_jigsaw=image_jigsaw)
        labels = Munch(label_classification=label_classification,
                        ab_channels=ab_channels,
                        pos_vec=pos_vec,
                        rot_vec=rot_vec)

        return images, labels
    
    
# EXAMPLE USAGE
# dataset = ClassificationDataset('data', split='train', transform=transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)