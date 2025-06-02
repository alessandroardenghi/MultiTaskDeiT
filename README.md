# Remaining Tasks:
## CODING TASKS
- Uploading pretrained weights and txt files
- Writing code to do full reconstruction
- writing readme
- overall cleaning each file (imports to clean)
- merge everything into main

## REPORT TASKS
- generare loss graphs (if any)
- select images to import in the model
- improve conclusions
- make report shorter
- add contributions more explicitly
- fix and motivate references

# MultiTaskDeiT 
A comparative framework evaluating single-task versus three-task joint training to measure performance gains.

## Table of Contents 
- [Introduction](#introduction) 
- [Paper](#paper) - [Installation](#installation) 
- [Prerequisites](#prerequisites) 
- [Conda Environment Setup](#conda-environment-setup) 
- [Dataset & Pretrained Models](#dataset--pretrained-models) 
- [Usage](#usage) 
- [Training](#training) 
- [Inference](#inference) 
- [Results](#results) 
- [Citation](#citation) 
- [License](#license) 

## Introduction (TO DO)

A brief overview of the project, its goals, and key contributions. You can mention high-level ideas, why your approach is novel, and what tasks or datasets it addresses. 


## Paper (TO DO)

If your project is associated with a publication, link it here: 


**Title:** [INSERT PAPER TITLE HERE](INSERT_PAPER_URL_HERE) 
**Authors:** First Author, Second Author, … 
**Conference/Journal:** Venue Name, Year 
&gt; **Abstract (optional):** 
&gt; A concise abstract or summary of the paper. 

## Installation 
### Prerequisites 
- Python 3.12+ (tested on 3.12.9) 
- CUDA 12.4 (optional, for GPU training/inference) 
- `git` 
- `conda` (Anaconda or Miniconda) 

### Conda Environment Setup 
1. Clone the repository:

   ```bash
   git clone https://github.com/alessandroardenghi/SolvingJigsawPuzzles.git
   cd SolvingJigsawPuzzles 
   ```
2. Create a new conda environment: ```bash conda create -n project-env python=3.9 -y ``` 
3. Activate the environment: ```bash conda activate project-env ``` 
4. Install required packages: ```bash conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # Example for PyTorch pip install -r requirements.txt ``` 

&gt; **Note:** Adjust the CUDA version and package manager flags as needed. 

5. Verify installation: ```bash python -c "import torch; print('CUDA available:', torch.cuda.is_available())" ```
--- 
## Dataset & Pretrained Models 
- **Dataset Links:** 
- Download link for training data: [INSERT DATA LINK HERE] 
- Instructions for preprocessing (if applicable) 
- **Pretrained Weights:** 
- Download link for pretrained model weights: [INSERT PRETRAINED WEIGHTS LINK HERE] 
- MD5/SHA256 checksums (optional): ``` pretrained_model.pth MD5: <insert_md5_here> ``` 
- **Directory Structure (after download/unpacking):** ``` project-name/ ├── data/ │ ├── train/ │ ├── val/ │ └── test/ ├── pretrained_models/ │ └── pretrained_model.pth └── ... ``` 
--- 
## Usage ### 
Training 
1. Configure training settings in `configs/train_config.yaml` (or specify hyperparameters via command line): ```yaml # Example YAML entries model: name: MyModel backbone: resnet50 training: batch_size: 32 lr: 0.001 num_epochs: 100 save_checkpoint: true ``` 
2. Run training script: ```bash python train.py \ --config configs/train_config.yaml \ --data_dir data/ \ --output_dir outputs/ \ --device cuda:0 ``` 
&gt; **Optional arguments:** 
&gt; `--resume <checkpoint_path>` to resume training from a checkpoint. 
&gt; `--seed <int>` to set a random seed for reproducibility. 
3. Logs and checkpoints will be saved in `outputs/`. Monitor training via TensorBoard: ```bash tensorboard --logdir outputs/logs/ ``` 
--- 
### Inference 
1. Prepare the inference configuration in `configs/infer_config.yaml`: ```yaml # Example YAML entries model: checkpoint: pretrained_models/pretrained_model.pth inference: batch_size: 16 input_size: 224 ``` 

2. Run inference script: ```bash python infer.py \ --config configs/infer_config.yaml \ --data_dir data/test/ \ --output_dir outputs/inference/ \ --device cuda:0 ``` 
3. Output predictions will be saved to `outputs/inference/`. You can post-process results, e.g., converting logits to labels: ```bash python postprocess.py \ --predictions outputs/inference/preds.npy \ --labels outputs/inference/labels.txt ``` 
--- 
## Results 
| Model | Dataset | Top-1 Accuracy | Top-5 Accuracy | mAP | Comments | 
|---------------------|--------------|----------------|----------------|----------|------------------------| 
| Baseline | Dataset A | 75.20% | 92.15% | 0.672 | ResNet-50 backbone | 
| **Our Proposed 🏆** | Dataset A | **78.45%** | **94.03%** | **0.703**| + Data augmentation | 
| Our Method (Variant)| Dataset A | 77.18% | 93.10% | 0.689 | Different hyperparams | 
| ... | ... | ... | ... | ... | ... | 
&gt; **Notes:** 
&gt; 
- Fill in this table with your own experimental results. 
&gt; 
- Add or remove columns as necessary (e.g., Precision, Recall, F1-score, etc.). 
&gt; 
- You can also include validation curves or example qualitative results (images) below.
--- 
## Citation 
If you find this code useful for your research, please cite our paper: ```bibtex @inproceedings{Author2025ProjectName, title = {Project Name: A Novel Approach to XYZ}, author = {First Author and Second Author and Third Author}, booktitle = {Proceedings of Conference Name}, year = {2025}, pages = {123--134}, doi = {10.xxxx/xxxxxxx} } ``` --- ## License Specify your license here. For example: ``` MIT License Copyright (c) 2025 Name Permission is hereby granted, free of charge, to any person obtaining a copy ... ``` 