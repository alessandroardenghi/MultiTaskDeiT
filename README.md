# Remaining Tasks:
## REPORT TASKS
- Grafici: model architecture, results, colorization results (best and worst)?
- Describe coloring loss
- classification metrics (aggiungere)
- Results da aggiungere. Aggiungere tables di results e coi parametri
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

## Paper 
**Title:** Self-Supervision through Image Reconstruction: Can Multitask Training Enhance Feature Representation?
**Authors:** Ardenghi Alessandro, Giampetruzzi Rocco Cristiano <br>
**Abstract:** <br>
In this paper, we investigate the effect of multi-task learning on the feature-representation capabilities of a Vision Transformer. <br>
We propose a model composed of a Vision Transformer backbone and three specialized heads, trained simultaneously to reconstruct jigsaw-puzzled grayscale images, classify their reconstructed versions, and colorize them. <br>
This setup is designed to encourage the model to learn high-level semantic features via puzzle solving and classification, while also capturing fine-grained pixel-level details through the coloring task. <br>
We hypothesize that this synergy could lead to more robust and generalizable latent image representations. <br>
Experiments conducted on the COCO and Pascal VOC datasets show that our multi-task approach achieves results comparable to training the tasks separately, with no significant performance gains. <br>
While these findings suggest that multi-task learning does not inherently improve performance in this context, we identify several design choices that may have limited the model’s effectiveness and outline directions for future improvements.

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
1. Configure classification inference settings in `configs/eval/config_class.yaml`

2. Run inference script: ```python3 -m eval_script/eval_class``` 
3. Output predictions will be saved to `model_results/class_top_metrics/{experiment_name}` 
--- 
### Inference Jigsaw Reconstruction
1. Configure jigsaw inference settings in `configs/eval/config_jigsaw.yaml`

2. Run inference script: ```python3 -m eval_script/eval_jigsaw``` 
3. Output predictions will be saved to `model_results/jigsaw_metrics/{experiment_name}` 
--- 
### Inference Colorization
1. Configure colorization inference settings in `configs/eval/config_coloring.yaml`

2. Run inference script: ```python3 -m eval_script/eval_coloring``` 
3. Colorized images will be saved to `model_results/coloring_results/{experiment_name}` 
--- 
## Results 
#### Classification Task

| Model     | Accuracy | Avg-F1 | Best-F1 |
|-----------|----------|--------|---------|
| Frozen    | 19.24    | 45.91  | 89.10   |
| Finetune  | 24.99    | 55.77  | 90.28   |
| Multi     | 23.91    | 54.84  | 90.35   |

#### Jigsaw Task

| Model     | Top1-Pos | Top3-Pos | Top1-Rot |
|-----------|----------|----------|----------|
| Random    | 11.11    | 33.33    | 25.00    |
| Frozen    | 21.63    | 55.03    | 37.06    |
| Finetune  | 33.11    | 76.13    | 55.64    |
| Multi     | 32.71    | 75.01    | 54.58    |
