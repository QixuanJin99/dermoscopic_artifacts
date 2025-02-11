import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
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
from torch.utils.data import DataLoader, Subset
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import pickle

parser = argparse.ArgumentParser(description='Train dermatology artifact classifier')
parser.add_argument('--curr_artifact', type=str, default="ruler")
args = parser.parse_args()

curr_artifact = args.curr_artifact
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_file = "/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv"  
image_dir = "/data/healthy-ml/gobi1/data/ISIC/2018_train_task1-2" 

save_dir = "/mnt/scratch-lids/scratch/qixuanj/chil2025/artifact_classifiers2"
eval_dir = "/mnt/scratch-lids/scratch/qixuanj/chil2025/artifact_classifiers2_eval"
df = pd.read_csv(csv_file, index_col=0)

artifact_labels = ["dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Lower prevalence classes
prevalence = df[artifact_labels].mean(axis=0)
class_weights = {artifact: 1.0 / prevalence[artifact] for artifact in ["gel_border", "ink", "patches"]}

# Convert to PyTorch tensor
for artifact, weight in class_weights.items():
    class_weights[artifact] = torch.tensor(weight, dtype=torch.float).to(DEVICE)

class ArtifactDataset(Dataset):
    def __init__(self, df, image_dir, target_label, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.target_label = target_label 
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row[self.target_label], dtype=torch.float32)  # Binary label (0 or 1)
        return image, label

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"epoch {epoch}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).squeeze()  # Ensure shape matches (batch_size,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


def test_model(model, test_loader, split):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images).squeeze()  # Ensure shape is (batch_size,)
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            preds = (probs > 0.5).float()  # Convert probabilities to binary predictions
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    auroc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else float('nan')
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)

    results = {
        "AUROC": auroc,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision
    }

    return results, {split: all_preds.tolist()}, {split: all_labels.tolist()}

for j in range(1, 6): 
    split_column = f"split_{j}"  # Change this based on the desired split

    # Create dataset
    dataset = ArtifactDataset(df, image_dir, target_label=curr_artifact, transform=transform)
    
    # Get indices for train and test split based on the column
    train_indices = df[df[split_column] == "train"].index.tolist()
    test_indices = df[df[split_column] == "test"].index.tolist()
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Pretrained ResNet-50 and Modify Final Layer
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Single neuron for binary classification
    model = model.to(DEVICE)

    # Define Loss Function and Optimizer
    if artifact in class_weights: 
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[artifact])
    else:  
        criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

    # Save the model
    model_path = f"{save_dir}/{curr_artifact}_classifier_split_{j}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model for {curr_artifact}: {model_path}\n")


print("Run evaluation")
# Run evaluation for each split
results_dict = {}
all_preds_dict = {}
all_labels_dict = {}

for j in range(1, 6): 
    split_column = f"split_{j}"

    # Load the trained model
    model = models.resnet50(pretrained=False)  # Define model architecture
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(DEVICE)

    model_path = f"{save_dir}/{curr_artifact}_classifier_split_{j}.pth"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f" Evaluating model for {curr_artifact}: {model_path}")


    # Run evaluation
    results, preds, labels = test_model(model, test_loader, split_column)

    # Store results
    results_dict[split_column] = results
    all_preds_dict.update(preds)
    all_labels_dict.update(labels)

with open (f"{eval_dir}/{curr_artifact}_results.pkl", "wb") as f: 
    pickle.dump(results_dict, f)

with open (f"{eval_dir}/{curr_artifact}_preds.pkl", "wb") as f: 
    pickle.dump(all_preds_dict, f)

with open (f"{eval_dir}/{curr_artifact}_labels.pkl", "wb") as f: 
    pickle.dump(all_labels_dict, f)
    
# Print evaluation results
for split, metrics in results_dict.items():
    print(f"📊 Results for {split}: {metrics}")
