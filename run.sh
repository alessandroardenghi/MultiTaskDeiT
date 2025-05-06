#!/bin/bash
#SBATCH --account=3141445
#SBATCH --partition=dsba
#SBATCH --gpus=3
#SBATCH --job-name=dataset
#SBATCH --mem=21G
#SBATCH --cpus-per-task=16
#SBATCH --error=err/%x_%j.err
#SBATCH --output=out/%x_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=3141445@studdbocconi.it


# Activate conda
module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate vit-env


# Run your script
time python main_training.py

# Deactivate conda
conda deactivate
module purge

