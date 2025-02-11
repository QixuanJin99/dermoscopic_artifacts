import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
import argparse
from datasets import ISICDataset, HAM10000Dataset, PH2Dataset, BCN20000Dataset

parser = argparse.ArgumentParser(description='Run artifact labels')
parser.add_argument('--dataset', type=str, default="PH2")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
dataset_mode = "whole"
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

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

if args.dataset == "ISIC": 
    df = pd.read_csv("/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv", index_col=0)
    full_dataset = ISICDataset(df, image_dir, mask_dir, transform=transform, mode=dataset_mode, return_pil=False)
elif args.dataset == "HAM10000":
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/HAM10000_metadata_with_objects.csv", index_col=0)
    full_dataset = HAM10000Dataset(df, image_dir, mask_dir, transform=transform, mode=dataset_mode, return_pil=False)
elif args.dataset == "PH2": 
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/ph2_metadata.csv", index_col=0)
    full_dataset = PH2Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False)
elif args.dataset == "BCN20000": 
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/bcn20000_metadata.csv", index_col=0)
    full_dataset = BCN20000Dataset(df, image_dir, transform=transform, mode=dataset_mode, return_pil=False)

artifacts = ["dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches"]
save_dir = "/mnt/scratch-lids/scratch/qixuanj/chil2025/artifact_classifiers2"


all_probs_dict = {}   # Stores probability values (sigmoid output)
all_preds_dict = {}   # Stores binary predictions (0 or 1)

for artifact in artifacts:
    print(artifact)
    all_probs_dict[artifact] = {}
    all_preds_dict[artifact] = {}
    
    for j in range(1, 6): 
        split_column = f"split_{j}"
        print(split_column)
    
        # Load the trained model
        model = models.resnet50(pretrained=False)  # Define model architecture
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
        model = model.to(DEVICE)
    
        model_path = f"{save_dir}/{artifact}_classifier_split_{j}.pth"
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # Prepare dataset for the current artifact split
        dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

        all_probs = []
        all_preds = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Inference {artifact} - Split {j}"):
                images = images.to(DEVICE)

                outputs = model(images).squeeze()  # Ensure shape is (batch_size,)
                probs = torch.sigmoid(outputs)  # Convert logits to probabilities
                preds = (probs > 0.5).float()  # Convert probabilities to binary predictions
                
                all_probs.extend(probs.cpu().numpy())   # Store probabilities
                all_preds.extend(preds.cpu().numpy())   # Store binary predictions

        # Store results in dictionaries
        all_probs_dict[artifact][split_column] = np.array(all_probs)
        all_preds_dict[artifact][split_column] = np.array(all_preds)


save_path = f"/mnt/scratch-lids/scratch/qixuanj/chil2025/artifact_preds2/{args.dataset}_artifact_predictions.pkl"
with open(save_path, "wb") as f:
    pickle.dump({"probs": all_probs_dict, "preds": all_preds_dict}, f)

print(f"Saved probabilities, binary predictions, and labels to {save_path}")