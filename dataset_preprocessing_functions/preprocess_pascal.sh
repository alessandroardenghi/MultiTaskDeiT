#!/bin/bash

#SBATCH -c 16
#SBATCH --mem=5G
#SBATCH -p dsba
#SBATCH -J setting_up_dir
#SBATCH -e _err/err-%j.err
#SBATCH -o _out/out-%j.out
#SBATCH --gpus=0

source /home/3144656/miniconda3/bin/activate

conda activate jigsaw

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