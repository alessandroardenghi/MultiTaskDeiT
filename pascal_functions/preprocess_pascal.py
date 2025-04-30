import os
import shutil
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Define paths
source_images_dir = "VOCdevkit/VOC2012/JPEGImages"
source_annotations_dir = "VOCdevkit/VOC2012/Annotations"
destination_images_dir = "data/images"
output_labels_file = "data/labels.npz"

# Create destination directory if it doesn't exist
os.makedirs(destination_images_dir, exist_ok=True)

# Initialize the dictionary to store image paths and labels
labels_dict = {}

# Get the list of image files
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(".jpg")]

# Process images and annotations with a progress bar
for image_file in tqdm(image_files, desc="Processing images"):
    # Move image to destination
    source_image_path = os.path.join(source_images_dir, image_file)
    destination_image_path = os.path.join(destination_images_dir, image_file)
    shutil.move(source_image_path, destination_image_path)

    # Parse corresponding annotation file
    annotation_file = os.path.join(source_annotations_dir, image_file.replace(".jpg", ".xml"))
    if os.path.exists(annotation_file):
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        # Extract labels (object names)
        labels = [obj.find("name").text for obj in root.findall("object")]
        labels_dict[destination_image_path] = labels

# Save the labels dictionary as a .npz file
os.makedirs("data", exist_ok=True)
np.savez(output_labels_file, **labels_dict)

print(f"Processed {len(labels_dict)} images and saved labels to {output_labels_file}.")