# Standard library imports
import os
import sys
import json
import pickle
import random
import re
from glob import glob
from pathlib import Path

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2
from tqdm import tqdm
import scipy

# PyTorch imports
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# torchvision imports
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import VisionDataset
import torchvision.transforms as T

# PIL imports
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, DataLoader
import scipy.ndimage

import torch.nn as nn
import torch.optim as optim
from torchvision import models
import argparse

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import torch.nn.functional as F

from datasets import ISICDataset, HAM10000Dataset, PH2Dataset, BCN20000Dataset


parser = argparse.ArgumentParser(description='Train trap dermatology classifier')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--dataset_mode', type=str, default="whole")
args = parser.parse_args()


# Define paths
image_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2"
mask_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2_segmentations"

# Create dataset instances for each mode
dataset_modes = ["whole", "lesion", "background", "bbox", "bbox70", "bbox90",
                 "high_whole", "low_whole", "high_lesion", "low_lesion", "high_background", "low_background"]

trap_sets = ['ruler_25', 'ruler_50', 'ruler_75', 'ruler_100', 
             'ink_25', 'ink_50', 'ink_75',  'ink_100', 
             'dark_corner_25',  'dark_corner_50', 'dark_corner_75', 'dark_corner_100']

df = pd.read_csv("/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv", index_col=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
num_epochs = args.num_epochs  # Adjust as needed
batch_size = 32
learning_rate = 1e-4

transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Select dataset mode (e.g., "whole", "lesion", etc.)
dataset_mode = args.dataset_mode  # Change this to your desired mode

# Create a directory to save models

for trap_set in trap_sets:
    save_dir = f"/mnt/scratch-lids/scratch/qixuanj/chil2025/classifiers_trap/{dataset_mode}/{trap_set}"
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = ISICDataset(df, image_dir, mask_dir, transform=transform, mode=dataset_mode, return_pil=False, trap_set = trap_set)
    print(trap_set)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )
    
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        print(f"Starting epoch {epoch}")
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # Ensure labels are (batch, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    model_path = f"{save_dir}/resnet50_split_1.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

print("Training completed for all splits!")