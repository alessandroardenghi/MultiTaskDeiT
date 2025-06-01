import os
import shutil
import json
import numpy as np
import cv2
from collections import defaultdict
import argparse

if __name__ == "__main__":
    # Paths
    
    parser = argparse.ArgumentParser(description="Split COCO images based on colorfulness and annotations.")
    parser.add_argument('--annotations_path', type=str, default='coco_data/annotations', help='Path to COCO annotations JSON')
    parser.add_argument('--output_dir', type=str, default='coco_data', help='Output COCO directory')
    args = parser.parse_args()
    
    coco_dir = args.output_dir
    images_src_dir = 'coco_data/val2014'                            # 'coco/images'  # Where COCO val images are currently
    images_dst_dir = os.path.join(coco_dir, 'images')
    annotations_path = args.annotations_path
    labels_file = os.path.join(coco_dir, 'labels.npz')

    os.makedirs(images_dst_dir, exist_ok=True)

    # Threshold for filtering out low-color images
    color_threshold = 2.0

    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Create mappings
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    filename_to_image_id = {img['file_name']: img['id'] for img in annotations['images']}
    categories = annotations['categories']
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}
    num_classes = len(categories)

    # images without annotations
    ann_by_image = defaultdict(list)
    for ann in annotations['annotations']:
        ann_by_image[ann['image_id']].append(ann)
    all_image_ids = [img['id'] for img in annotations['images']]
    images_without_anns = [img_id for img_id in all_image_ids if img_id not in ann_by_image]

    # Track valid images
    valid_images = set()
    #image_labels = defaultdict(lambda: np.zeros(num_classes, dtype=int))
    skipped = 0

    # Step 1: Filter and move images
    for fname in os.listdir(images_src_dir):
        if not fname.endswith('.jpg'):
            continue

        src_path = os.path.join(images_src_dir, fname)
        img_id = filename_to_image_id.get(fname)

        if img_id in images_without_anns:
            skipped += 1
            print(f"Skipping image without annotations: {fname}")
            continue

        try:
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                print(f"Failed to read {fname}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0

            # Convert to Lab
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            a_channel = img_lab[:, :, 1].astype(np.float32)
            b_channel = img_lab[:, :, 2].astype(np.float32)
            colorfulness = (np.std(a_channel) + np.std(b_channel)) / 2

            if colorfulness < color_threshold:
                skipped += 1
                print(f"Skipping low-color image: {fname} (colorfulness={colorfulness:.2f})")
                continue

            shutil.move(src_path, os.path.join(images_dst_dir, fname))
            valid_images.add(fname)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print(f"Skipped {skipped} low-color images.")


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
