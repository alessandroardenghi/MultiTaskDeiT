import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

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
    
    
    
# EXAMPLE USAGE
# dataset = ClassificationDataset('data', split='train', transform=transform)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)