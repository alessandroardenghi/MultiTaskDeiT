#!/bin/bash

# Example shell script to call multiple Python scripts

python3 dataset_preprocessing_functions/PascalVOC/download_pascal.py

python3 dataset_preprocessing_functions/PascalVOC/preprocess_pascal.py \
    --destination_dir pascal_data 

python3 dataset_preprocessing_functions/build_train_test.py pascal_data/images \
    --output_dir pascal_data 

if [ $? -ne 0 ]; then
    echo "A script failed to execute."
    exit 1
fi
rm -f VOCtrainval_11-May-2012.tar
rm -rf VOCdevkit
echo "PascalVOC was correctly downloaded and preprocessed."