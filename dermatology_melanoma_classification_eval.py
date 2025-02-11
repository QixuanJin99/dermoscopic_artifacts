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
from datasets import ISICDataset, HAM10000Dataset, PH2Dataset, BCN20000Dataset
import argparse
import pickle

import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

parser = argparse.ArgumentParser(description='Run evaluation')
parser.add_argument('--dataset', type=str, default="ISIC")
parser.add_argument('--augmented', type=str, default=None)
parser.add_argument('--clean', type=str, default=None)
args = parser.parse_args()

print(f"{args.dataset}")

# dataset_modes = ["whole", "lesion", "background", "bbox", "bbox70", "bbox90",
#                  "high_whole", "low_whole", "high_lesion", "low_lesion", "high_background", "low_background"]

# Dataset modes for PH2 inference augmented 
dataset_modes = ["whole", "background", "bbox", "bbox70", "bbox90",
                 "high_whole", "low_whole", "high_background", "low_background"]


if args.dataset == "ISIC": 
    image_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2"
    mask_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2_segmentations"
elif args.dataset == "HAM10000": 
    image_dir = "/data/healthy-ml/gobi1/data/ham10000/HAM10000_images"
    mask_dir = "/data/healthy-ml/gobi1/data/ham10000/HAM10000_segmentations"
elif args.dataset == "PH2": 
    image_dir = "/mnt/scratch-lids/data/PH2Dataset/images/"
elif args.dataset == "BCN20000":
    image_dir = "/mnt/scratch-lids/data/bcn20000/"
    

for dataset_mode in dataset_modes: 
    print(dataset_mode)
    all_metrics = {
        "AUROC": [],
        "Accuracy": [],
        "Recall": [],
        "Precision": []
    }
    # Directory containing saved models
    save_dir = f"/mnt/scratch-lids/scratch/qixuanj/chil2025/classifiers_ph2/{dataset_mode}"
    # save_dir = f"/mnt/scratch-lids/scratch/qixuanj/chil2025/classifiers/{dataset_mode}"

    if args.dataset == "ISIC": 
        df = pd.read_csv("/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv", index_col=0)
        full_dataset = ISICDataset(df, image_dir, mask_dir, transform=transform, mode=dataset_mode, return_pil=False)
    elif args.dataset == "HAM10000":
        df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/HAM10000_metadata_with_objects.csv", index_col=0)
        full_dataset = HAM10000Dataset(df, image_dir, mask_dir, transform=transform, mode=dataset_mode, return_pil=False)
    elif args.dataset == "PH2": 
        df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/ph2_metadata.csv", index_col=0)
        if args.augmented:
            full_dataset = PH2Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False, 
                                      augmented=args.augmented)
        elif args.clean: 
            full_dataset = PH2Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False, 
                                      clean=args.clean)
        else:
            full_dataset = PH2Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False)
    elif args.dataset == "BCN20000": 
        df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/bcn20000_metadata.csv", index_col=0)
        full_dataset = BCN20000Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False)
    
        
    store_preds = {}
    store_labels = {}
    
    # Loop through each split
    for split in range(1, 6):
        if 'high' in dataset_mode and split == 5: 
            continue 
        print(f"Evaluating {dataset_mode} - Split {split}")
    
        # Get test indices
        test_indices = df[df[f"split_{split}"] == "test"].index.tolist()
    
        # Create test dataset and DataLoader
        test_dataset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # Load model
        model = models.resnet50(pretrained=False)  # Load model architecture
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        
        # model.load_state_dict(torch.load(f"{save_dir}/resnet50_split_{split}.pth"))
        model.load_state_dict(torch.load(f"{save_dir}/resnet50_split{split}.pth"))
        model = model.to(device)
        model.eval()
    
        # Lists to store predictions and labels
        all_preds = []
        all_labels = []
    
        # Evaluation loop
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Evaluating Split {split}"):
                images = images.to(device)
                labels = labels.cpu().numpy()  # Convert labels to NumPy array
    
                outputs = model(images).cpu().numpy()  # Get model predictions
                preds = outputs.flatten()  # Flatten predictions
    
                all_preds.extend(preds)
                all_labels.extend(labels)
    
        # Convert lists to NumPy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        # Compute metrics
        auroc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds >= 0.5)
        recall = recall_score(all_labels, all_preds >= 0.5)
        precision = precision_score(all_labels, all_preds >= 0.5)
    
        # Store metrics
        all_metrics["AUROC"].append(auroc)
        all_metrics["Accuracy"].append(acc)
        all_metrics["Recall"].append(recall)
        all_metrics["Precision"].append(precision)
    
        store_preds[split] = all_preds 
        store_labels[split] = all_labels
    
        print(f"Split {split} - AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Compute mean metrics across splits
    mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    # Print final results
    print("\n===== Final Evaluation Results =====")
    for metric, mean_value in mean_metrics.items():
        print(f"Mean {metric}: {mean_value:.4f}")


    if args.dataset == "ISIC": 
        with open (f"{save_dir}/all_metrics.pkl", "wb") as f: 
            pickle.dump(all_metrics, f)
        with open (f"{save_dir}/store_preds.pkl", "wb") as f: 
            pickle.dump(store_preds, f)
        with open (f"{save_dir}/store_labels.pkl", "wb") as f: 
            pickle.dump(store_labels, f)
    elif args.dataset == "HAM10000":
        with open (f"{save_dir}/all_metrics_ham10000.pkl", "wb") as f: 
            pickle.dump(all_metrics, f)
        with open (f"{save_dir}/store_preds_ham10000.pkl", "wb") as f: 
            pickle.dump(store_preds, f)
        with open (f"{save_dir}/store_labels_ham10000.pkl", "wb") as f: 
            pickle.dump(store_labels, f)
    elif args.dataset == "PH2":
        if args.augmented:
            with open (f"{save_dir}/all_metrics_ph2_augmented_{args.augmented}.pkl", "wb") as f: 
                pickle.dump(all_metrics, f)
            with open (f"{save_dir}/store_preds_ph2_augmented_{args.augmented}.pkl", "wb") as f: 
                pickle.dump(store_preds, f)
            with open (f"{save_dir}/store_labels_ph2_augmented_{args.augmented}.pkl", "wb") as f: 
                pickle.dump(store_labels, f)
        elif args.clean: 
            with open (f"{save_dir}/all_metrics_ph2_clean_{args.clean}.pkl", "wb") as f: 
                pickle.dump(all_metrics, f)
            with open (f"{save_dir}/store_preds_ph2_clean_{args.clean}.pkl", "wb") as f: 
                pickle.dump(store_preds, f)
            with open (f"{save_dir}/store_labels_ph2_clean_{args.clean}.pkl", "wb") as f: 
                pickle.dump(store_labels, f)
        else:
            with open (f"{save_dir}/all_metrics_ph2.pkl", "wb") as f: 
                pickle.dump(all_metrics, f)
            with open (f"{save_dir}/store_preds_ph2.pkl", "wb") as f: 
                pickle.dump(store_preds, f)
            with open (f"{save_dir}/store_labels_ph2.pkl", "wb") as f: 
                pickle.dump(store_labels, f)
    elif args.dataset == "BCN20000":
        with open (f"{save_dir}/all_metrics_bcn20000.pkl", "wb") as f: 
            pickle.dump(all_metrics, f)
        with open (f"{save_dir}/store_preds_bcn20000.pkl", "wb") as f: 
            pickle.dump(store_preds, f)
        with open (f"{save_dir}/store_labels_bcn20000.pkl", "wb") as f: 
            pickle.dump(store_labels, f)