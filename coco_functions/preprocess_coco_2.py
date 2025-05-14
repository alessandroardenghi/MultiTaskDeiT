import os
import shutil
import json
import numpy as np
import pickle
from PIL import Image
import cv2
from collections import defaultdict

if __name__ == "__main__":
    # Paths
    coco_dir = 'coco_colors'                                   # 'coco/images'  # Where COCO val images are currently
    annotations_path = 'annotations/instances_val2014.json'
    labels_file = os.path.join(coco_dir, 'labels.npz')

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create mappings
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    categories = annotations['categories']
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
    num_classes = len(categories)

    # Track valid images
    valid_images = set()
    image_labels = defaultdict(lambda: np.zeros(num_classes, dtype=int))
    skipped = 0

    # Step 2: Assign one-hot labels only for valid images
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        fname = image_id_to_filename.get(img_id)
        if fname in valid_images:
            cat_id = ann['category_id']
            index = cat_id_to_index[cat_id]
            image_labels[fname][index] = 1

    np.savez(labels_file, labels=dict(image_labels), classes=categories)