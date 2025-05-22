import os
import json
import numpy as np
from collections import defaultdict


annotation_path = 'annotations/instances_val2014.json'
with open(annotation_path, 'r') as f:
    cocolabels = json.load(f)
# Create a map of image_id -> list of annotations
filename_to_image_id = {img['file_name']: img['id'] for img in cocolabels['images']}
ann_by_image = defaultdict(list)
for ann in cocolabels['annotations']:
    ann_by_image[ann['image_id']].append(ann)
    if ann['image_id'] == filename_to_image_id["COCO_val2014_000000087070.jpg"]:
        print("Image with annotations found:", filename_to_image_id["COCO_val2014_000000087070.jpg"])
        print(ann)

# List all image IDs in the dataset
all_image_ids = [img['id'] for img in cocolabels['images']]

# Find images without annotations
images_without_anns = [img_id for img_id in all_image_ids if img_id not in ann_by_image]

# if filename_to_image_id["COCO_val2014_000000087070.jpg"] in images_without_anns:
#     print("Image without annotations found:", filename_to_image_id["COCO_val2014_000000087070.jpg"])
# else:
#     print("Image with annotations found:", filename_to_image_id["COCO_val2014_000000087070.jpg"])
#     print(cocolabels['annotations'])


print(f"{len(images_without_anns)} over {len(all_image_ids)} images without annotations")