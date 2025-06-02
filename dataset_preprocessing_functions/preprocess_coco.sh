#!/bin/bash

python3 'dataset_preprocessing_functions/COCO/download_coco.py'

rm -rf coco_data/annotations_trainval2014.zip
rm -rf coco_data/val2014.zip


python3 dataset_preprocessing_functions/COCO/preprocess_coco.py \
     --output_dir coco_data \
     --annotations_path coco_data/annotations/instances_val2014.json

rm -rf coco_data/val2014
rm -rf coco_data/annotations

python3 dataset_preprocessing_functions/build_train_test.py coco_data/images \
    --output_dir coco_data 

if [ $? -ne 0 ]; then
    echo "A script failed to execute."
    exit 1
fi

echo "COCO was correctly downloaded and preprocessed."