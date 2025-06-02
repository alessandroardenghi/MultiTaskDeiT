# Remaining Tasks:
## REPORT TASKS
- Grafici: model architecture, results, colorization results (best and worst)?
- Describe coloring loss
- classification metrics (aggiungere)
- Results da aggiungere. Aggiungere tables di results e coi parametri
- Abstract
- select images to import in the model
- improve conclusions
- make report shorter
- add contributions more explicitly
- fix and motivate references

# MultiTaskDeiT 
A comparative framework evaluating single-task versus three-task joint training to measure performance gains.

## Table of Contents 
- [Introduction](#introduction) 
- [Paper](#paper) 
- [Installation](#installation) 
- [Prerequisites](#prerequisites) 
- [Conda Environment Setup](#conda-environment-setup) 
- [Dataset & Pretrained Models](#dataset--pretrained-models) 
- [Usage](#usage) 
- [Training](#training) 
- [Inference](#inference) 
- [Results](#results) 

## Introduction

Previous papers found that adding self-supervised tasks to the training of classifier models improved the feature representation capabilities of the model itself, in turn improving the classification results.
The objective of this project is therefore to investigate the effects of training a model to perform classification, while also learning to perform Jigsaw Reconstruction and Image Colorization. 

## Paper (TO DO)
**Title:** Self-Supervision through Image Reconstruction: Can Multitask Training Enhance Feature Representation?
**Authors:** Ardenghi Alessandro, Giampetruzzi Rocco Cristiano <br>
**Abstract:** A concise abstract or summary of the paper.

## Installation 
### Prerequisites 
- Python 3.12+ (tested on 3.12.9) 
- CUDA 12.4 (optional, for GPU training/inference) 
- `git` 
- `conda` (Anaconda or Miniconda) 

### Conda Environment Setup 
1. Clone the repository:

   ```bash
   git clone https://github.com/alessandroardenghi/MultiTaskDeiT.git
   cd MultiTaskDeiT
   ```
2. Create a new conda environment: 

    ```bash 
    conda env create -f requirements/environment.yml
    ``` 

3. Activate the environment: 
    
    ```bash 
    conda activate MultiTaskDeiT
    ``` 

4. Install required packages:  
Choose one of the following based on your system:
- Option 1: CPU-only (recommended for machines without a GPU)
    ```bash 
    pip install -r requirements/requirements_cpu.txt
    ``` 
- Option 2: GPU (requires compatible NVIDIA GPU + CUDA drivers)
    ```bash 
    pip install -r requirements/requirements_gpu.txt
    ``` 
--- 
## Dataset & Pretrained Models 
**Dataset Links:**  
Choose one of the following options (the following scripts will download, preprocess and split into train/val/test the chosen dataset):
- Option 1: PascalVOC Dataset
    ```bash 
    ./dataset_preprocessing_functions/preprocess_pascal.sh
    ``` 
- Option 2: PascalVOC Dataset
    ```bash 
    ./dataset_preprocessing_functions/preprocess_coco.sh
    ``` 
**Pretrained Weights:** 
- Download link for pretrained model weights:
   -  [MultiTaskDeiT Model Weights](https://github.com/alessandroardenghi/MultiTaskDeiT/releases/download/v1.0/MultiTaskDeiT.pth)
   -  [Single-Head Colorization Model Weights](https://github.com/alessandroardenghi/MultiTaskDeiT/releases/download/v1.0.0/coloring_single_head.pth)
   -  [Single-Head Jigsaw Reconstruction Model Weights](https://github.com/alessandroardenghi/MultiTaskDeiT/releases/download/v1.0.1/jigsaw_3x3_single_head.pth)
   -  [Single-Head Classification Model Weights](https://github.com/alessandroardenghi/MultiTaskDeiT/releases/download/v1.0.2/classification_single_head.pth)



**Train, Test Splits:** 
- The train, val and test splits used to train and evaluate the pretrained models on COCO are provieded in the __precomputed_splits__ directory. To replicate our experimental results, substitute the newly generated train.txt, test.txt, val.txt with the ones present in the __precomputed_splits__ directory.

**Directory Structure (example with PascalVOC):** 
``` 
MultiTaskDeit/
├── pascal_data/
│   ├── images/
│   ├── labels.npz
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── pretrained_models/  # OPTIONAL
│   ├── classification_single_head.pth
│   ├── coloring_single_head.pth
│   ├── jigsaw_3x3_single_head.pth
│   └── MultiTaskDeiT.pth
...
``` 
--- 
## Usage ### 
### Training Models from Frozen Backbone
1. Configure training settings in `configs/train/config.yaml`. (It is possible to train models with 1, 2 or 3 heads)
2. Run training script: ```python3 train.py```
3. Logs and checkpoints will be saved in `logs/{experiment_name}`.
--- 
### Finetuning MultiTaskDeiT (3-head model)
1. Configure training settings in `configs/train/config_multi.yaml` (Paths to the three pretrained heads should be included here)
2. Run training script: ```python3 train_multi.py```
3. Logs and checkpoints will be saved in `logs/{experiment_name}`.
--- 
### Inference Classification
1. Configure classification inference settings in `configs/eval/configs/eval/config_class.yaml`

2. Run inference script: ```python3 eval_class.py``` 
3. Output predictions will be saved to `model_results/class_top_metrics/{experiment_name}` 
--- 
### Inference Jigsaw Reconstruction
1. Configure jigsaw inference settings in `configs/eval/configs/eval/config_jigsaw.yaml`

2. Run inference script: ```python3 eval_jigsaw.py``` 
3. Output predictions will be saved to `model_results/jigsaw_metrics/{experiment_name}` 
--- 
### Inference Colorization
1. Configure colorization inference settings in `configs/eval/configs/eval/config_coloring.yaml`

2. Run inference script: ```python3 eval_coloring.py``` 
3. Colorized images will be saved to `model_results/coloring_results/{experiment_name}` 
--- 
## Results 
| Model | Dataset | Top-1 Accuracy | Top-5 Accuracy | mAP | Comments | 
|---------------------|--------------|----------------|----------------|----------|------------------------| 
| Baseline | Dataset A | 75.20% | 92.15% | 0.672 | ResNet-50 backbone | 
| **Our Proposed 🏆** | Dataset A | **78.45%** | **94.03%** | **0.703**| + Data augmentation | 
| Our Method (Variant)| Dataset A | 77.18% | 93.10% | 0.689 | Different hyperparams | 
| ... | ... | ... | ... | ... | ... | 
--- 
