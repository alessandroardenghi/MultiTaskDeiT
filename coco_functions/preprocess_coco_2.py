import os
import json
import numpy as np
from collections import defaultdict

if __name__ == "__main__":
    # Paths
    coco_dir = 'coco_colors'                                   
    annotations_path = 'annotations/instances_val2014.json'
    images_dir = os.path.join(coco_dir, 'images')          
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
    valid_images = {
                    filename
                    for filename in os.listdir(images_dir)
                    if os.path.isfile(os.path.join(images_dir, filename))
                    }

    image_labels = {}
    # Assign one-hot labels only for valid images
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        fname = image_id_to_filename.get(img_id)
        if fname in valid_images: 
            cat_id = ann['category_id']
            index = cat_id_to_index[cat_id]
            if fname in image_labels:
                image_labels[fname][index] = 1.0
            else:
                image_labels[fname] = np.zeros(num_classes, dtype=np.float32)
                image_labels[fname][index] = 1.0

    # Save as .npz with proper structure
    np.savez(labels_file, labels=image_labels, classes=categories)

    