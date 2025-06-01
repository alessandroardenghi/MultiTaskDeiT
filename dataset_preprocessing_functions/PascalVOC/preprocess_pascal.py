import os
import shutil
import numpy as np
from tqdm import tqdm
import argparse
import xml.etree.ElementTree as ET

# Define paths
parser = argparse.ArgumentParser(description="Preprocess Pascal VOC dataset.")
parser.add_argument("--destination_dir", type=str, default="data", help="Path to destination images directory")
args = parser.parse_args()

destination_dir = args.destination_dir

source_images_dir = "VOCdevkit/VOC2012/JPEGImages"
source_annotations_dir = "VOCdevkit/VOC2012/Annotations"
destination_images_dir = os.path.join(destination_dir, "images")
output_labels_file = os.path.join(destination_dir, "labels.npz")


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class_to_idx = {cls: i for i, cls in enumerate(VOC_CLASSES)}


# Create destination directory if it doesn't exist
os.makedirs(destination_images_dir, exist_ok=True)

# Initialize the dictionary to store image paths and labels
labels_dict = {}

# Get the list of image files
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(".jpg")]

labels_dict = {}

for image_file in tqdm(image_files, desc="Processing images"):
    shutil.copy(
        os.path.join(source_images_dir, image_file),
        os.path.join(destination_images_dir, image_file)
    )
    annotation_file = os.path.join(source_annotations_dir, image_file.replace(".jpg", ".xml"))
    if os.path.exists(annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        labels = [obj.find("name").text for obj in root.findall("object")]
        one_hot = np.zeros(len(VOC_CLASSES), dtype=np.float32)
        for label in labels:
            if label in class_to_idx:
                one_hot[class_to_idx[label]] = 1.0
        labels_dict[image_file] = one_hot

np.savez(output_labels_file, labels=labels_dict, classes=VOC_CLASSES)

print(f"Processed {len(labels_dict)} images and saved labels to {output_labels_file}.")