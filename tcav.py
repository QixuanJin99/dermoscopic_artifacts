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
from datasets import ISICDataset, HAM10000Dataset, PH2Dataset, BCN20000Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Subset, DataLoader
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import joblib


parser = argparse.ArgumentParser(description='Run TCAV evaluation')
parser.add_argument('--dataset', type=str, default="ISIC")
parser.add_argument('--dataset_mode', type=str, default="whole")
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

print(f"Dataset mode: {args.dataset_mode}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


save_dir = f"/mnt/scratch-lids/scratch/qixuanj/chil2025/classifiers/{args.dataset_mode}"

model = models.resnet50(pretrained=False) 
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(f"{save_dir}/resnet50_split_{args.split}.pth"))
model = model.to(device)
model.eval()

artifact_columns = ["dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches"]

if args.dataset == "ISIC": 
    df = pd.read_csv("/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv")
    image_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2"
    mask_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2_segmentations"
    full_dataset = ISICDataset(df, image_dir, mask_dir, transform=transform, mode=args.dataset_mode, return_pil=False)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

def get_activations(img):
    with torch.no_grad():
        model(img)
    feature_map = activations["features"].cpu().numpy()  # Shape: (batch_size, C, H, W)
    feature_vector = feature_map.reshape(feature_map.shape[0], -1)  # Flatten per sample
    return feature_vector 

def hook_fn(module, input, output):
    activations["features"] = output.detach()

def compute_tcav_score(concept_vector, activations):
    tcav_scores = np.dot(activations, concept_vector.T) > 0  # Compute alignment
    return np.mean(tcav_scores)  # Fraction of test images influenced by concept
    
# Choose the convolutional layer to extract features
activation_layers = ["layer3", "layer4"]  

for activation_layer in activation_layers:
    # Hook to capture activations
    activations = {}
    
    # Register hook
    layer = dict([*model.named_modules()])[activation_layer]
    layer.register_forward_hook(hook_fn)

    all_activations = []
    for batch in tqdm(full_loader): 
        img = batch[0].to(device)
        all_activations.append(get_activations(img)) 
    all_activations = np.vstack(all_activations)
    os.makedirs(f"/mnt/scratch-lids/scratch/qixuanj/chil2025/tcav/{args.dataset_mode}", exist_ok=True)
    np.save(f"/mnt/scratch-lids/scratch/qixuanj/chil2025/tcav/{args.dataset_mode}/all_activations_split{args.split}_{activation_layer}.npy", all_activations)

    benign_activations = all_activations[df[df['label']==0].index]
    malignant_activations = all_activations[df[df['label']==1].index]
    no_artifact_index = df[(df[artifact_columns] == 0).all(axis=1)].index
    random_activations =  all_activations[no_artifact_index]

    df_results = pd.DataFrame(index=artifact_columns, columns=["Benign_TCAV", "Malignant_TCAV"])
    for artifact in artifact_columns: 
        concept_activations = all_activations[df[df[artifact] == 1].index]
        X = np.concatenate([concept_activations, random_activations])
        y = np.concatenate([np.ones(len(concept_activations)), np.zeros(len(random_activations))])  # 1 = concept, 0 = random
        
        # Train linear classifier
        X = X.reshape(X.shape[0], -1)  # Flatten feature vectors
        cav_classifier = SGDClassifier(alpha=0.01, max_iter=1000)
        cav_classifier.fit(X, y)
        
        # Get Concept Activation Vector (CAV)
        cav_vector = cav_classifier.coef_
        
        benign_tcav_score = compute_tcav_score(cav_vector, benign_activations)
        malignant_tcav_score = compute_tcav_score(cav_vector, malignant_activations)
        df_results.loc[artifact] = {
            "Benign_TCAV": benign_tcav_score,
            "Malignant_TCAV": malignant_tcav_score
        }

        # Store results in DataFrame
    df_results.to_csv(f"/mnt/scratch-lids/scratch/qixuanj/chil2025/tcav/{args.dataset_mode}/tcav_scores_split{args.split}_{activation_layer}.csv")