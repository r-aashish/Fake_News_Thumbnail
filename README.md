# Reproducibility Study of "How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image", ACL, 2022

This repository contains the code and data used to reproduce the results of the paper "How does fake news use a thumbnail? CLIP-based Multimodal Detection on the Unrepresentative News Image" published at ACL 2022.

We have implemented various approaches including VILT, ClipScore, Clip Classifier, and Multilingual to detect the similarity between the news title T and its corresponding thumbnail image I. The necessary datasets are provided in the Datasets folder, and the instructions for running the code are included below.

# Task
The objective is to identify news articles with unrepresentative thumbnail images, given a news title and thumbnail image. The binary ground truth label indicates whether the thumbnail image represents the news title or not.

## Requirements
- Python 3.6 or higher
- numpy
- pandas
- torch
- torchvision 
- pillow 
- tqdm 
- transformers
- requests
- timm

## Baseline
Using the CLIP text and visual embeddings, we constructed a baseline model that forecasts the binary label on the thumbnail representativeness using the cosine similarity. The command listed below can be used to train the model.We Provide the datasets for English and the multilinguality in the datasets folder, but you can create your own datasets using the suggested method in the Paper that is in the Repo.

``` python
python CLIP_classifier.py  --train_path /path/to/train/dataset \
                            --val_path /path/to/validation/dataset  \
                            --batch_size 128 \
                            --learning_rate 0.001 \
                            --num_epochs 10 \
                            --max_grad_norm 1.0 \
                            --seed 1337 \
                            --sched_mode min \
                            --factor 0.5 \
                            --test_path /path/to/test/dataset  \
                            --patience 1 \
                            --save model_pt \
                            --sa_num 1
```
