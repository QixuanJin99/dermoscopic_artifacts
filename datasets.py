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
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage

def high_pass_filter(image, sigma=1, grayscale=False):
    """
    Apply a high-pass filter to an image.

    Args:
        image (numpy.ndarray): Input image in RGB format.
        sigma (float): Standard deviation for Gaussian blur.
        grayscale (bool): If True, converts image to grayscale before filtering.

    Returns:
        numpy.ndarray: High-pass filtered image.
    """
    if grayscale:
        # Convert image to grayscale before filtering (avoids color artifacts)
        image_gray = np.dot(image[..., :3], [0.2989, 0.587, 0.114])  # Convert to grayscale
        low_frequencies = scipy.ndimage.gaussian_filter(image_gray, sigma=sigma)
        high_frequencies = image_gray - low_frequencies
        return np.stack([high_frequencies] * 3, axis=-1)  # Expand back to 3 channels for visualization

    else:
        # Apply filter to each RGB channel separately
        high_frequencies = np.zeros_like(image, dtype=np.float32)
        for c in range(3):  # Iterate over RGB channels
            low_frequencies = scipy.ndimage.gaussian_filter(image[:, :, c], sigma=sigma)
            high_frequencies[:, :, c] = image[:, :, c] - low_frequencies
        
        return high_frequencies

def low_pass_filter(image, sigma=1):
    return scipy.ndimage.gaussian_filter(image, sigma=sigma)

class ISICDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, transform=None, mode="whole", return_pil=False, trap_set =None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image names and labels.
            image_dir (str): Directory containing original images.
            mask_dir (str): Directory containing ground truth segmentations.
            transform (callable, optional): Optional transform to apply to images.
            mode (str): One of "whole", "lesion", "background", "bbox", "bbox70", 
                        "bbox90", "high_whole", "low_whole", "high_lesion",
                        "low_lesion", "high_background", "low_background".
        """
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.return_pil = return_pil
        self.trap_set = trap_set 

        # Subset to trap training set 
        if self.trap_set: 
            self.df = self.df[self.df[f"split_1_{self.trap_set}"] == 1]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_name = self.df.iloc[idx]['image']
        label = self.df.iloc[idx]['label']
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".JPG", "_segmentation.png"))
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load segmentation mask
        
        # Ensure images and masks are the same size
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 0).astype(np.uint8)
        
        if self.mode == "whole":
            processed_image = image
        
        elif self.mode == "lesion":
            processed_image = image * mask[:, :, np.newaxis]

        elif self.mode == "background":
            processed_image = image * (1 - mask[:, :, np.newaxis])

        elif self.mode in ["bbox", "bbox70", "bbox90"]:
            # Compute bounding box around lesion
            y_idxs, x_idxs = np.where(mask > 0)
            if len(y_idxs) == 0 or len(x_idxs) == 0:  # If no lesion
                processed_image = image * 0  # Blackout image
            else:
                y_min, y_max = y_idxs.min(), y_idxs.max()
                x_min, x_max = x_idxs.min(), x_idxs.max()
                
                # Compute the original bbox (for `bbox`)
                if self.mode == "bbox":
                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

                # Expand bbox for bbox70 and bbox90
                else:
                    expand_ratio = 0.7 if self.mode == "bbox70" else 0.9

                    img_h, img_w = image.shape[:2]
                    bbox_h = y_max - y_min
                    bbox_w = x_max - x_min

                    # Calculate expansion to reach desired percentage of total image
                    target_area = expand_ratio * img_h * img_w
                    bbox_center_y, bbox_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                    
                    # Compute new bbox size
                    new_bbox_h = int(np.sqrt(target_area * (bbox_h / bbox_w)))  # Keep aspect ratio
                    new_bbox_w = int(np.sqrt(target_area * (bbox_w / bbox_h)))

                    # Ensure it fits within image boundaries
                    y_min = max(0, bbox_center_y - new_bbox_h // 2)
                    y_max = min(img_h, bbox_center_y + new_bbox_h // 2)
                    x_min = max(0, bbox_center_x - new_bbox_w // 2)
                    x_max = min(img_w, bbox_center_x + new_bbox_w // 2)

                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

        elif self.mode.startswith("high_") or self.mode.startswith("low_"):
            base_image = None

            if "whole" in self.mode:
                base_image = image
            elif "lesion" in self.mode:
                base_image = image * mask[:, :, np.newaxis]
            elif "background" in self.mode:
                base_image = image * (1 - mask[:, :, np.newaxis])

            if base_image is not None:
                if "high_" in self.mode:
                    # processed_image = high_pass_filter(base_image)
                    processed_image = high_pass_filter(base_image, sigma=1, grayscale=True)
                else:
                    processed_image = low_pass_filter(base_image, sigma=1)
                    
        if self.return_pil:
            processed_image = Image.fromarray(processed_image.astype(np.uint8))
        else:
            if self.transform: 
                processed_image = Image.fromarray(processed_image.astype(np.uint8))
                processed_image = self.transform(processed_image)
        label = torch.tensor(label, dtype=torch.long)
        
        return processed_image, label

class HAM10000Dataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, transform=None, mode="whole", return_pil=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image names and labels.
            image_dir (str): Directory containing original images.
            mask_dir (str): Directory containing ground truth segmentations.
            transform (callable, optional): Optional transform to apply to images.
            mode (str): One of "whole", "lesion", "background", "bbox", "bbox70", 
                        "bbox90", "high_whole", "low_whole", "high_lesion",
                        "low_lesion", "high_background", "low_background".
        """
        self.df = df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.return_pil = return_pil
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_name = self.df.iloc[idx]['image_id']
        label = self.df.iloc[idx]['label']
        
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_name}_segmentation.png")
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load segmentation mask
        
        # Ensure images and masks are the same size
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 0).astype(np.uint8)
        
        if self.mode == "whole":
            processed_image = image
        
        elif self.mode == "lesion":
            processed_image = image * mask[:, :, np.newaxis]

        elif self.mode == "background":
            processed_image = image * (1 - mask[:, :, np.newaxis])

        elif self.mode in ["bbox", "bbox70", "bbox90"]:
            # Compute bounding box around lesion
            y_idxs, x_idxs = np.where(mask > 0)
            if len(y_idxs) == 0 or len(x_idxs) == 0:  # If no lesion
                processed_image = image * 0  # Blackout image
            else:
                y_min, y_max = y_idxs.min(), y_idxs.max()
                x_min, x_max = x_idxs.min(), x_idxs.max()
                
                # Compute the original bbox (for `bbox`)
                if self.mode == "bbox":
                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

                # Expand bbox for bbox70 and bbox90
                else:
                    expand_ratio = 0.7 if self.mode == "bbox70" else 0.9

                    img_h, img_w = image.shape[:2]
                    bbox_h = y_max - y_min
                    bbox_w = x_max - x_min

                    # Calculate expansion to reach desired percentage of total image
                    target_area = expand_ratio * img_h * img_w
                    bbox_center_y, bbox_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                    
                    # Compute new bbox size
                    new_bbox_h = int(np.sqrt(target_area * (bbox_h / bbox_w)))  # Keep aspect ratio
                    new_bbox_w = int(np.sqrt(target_area * (bbox_w / bbox_h)))

                    # Ensure it fits within image boundaries
                    y_min = max(0, bbox_center_y - new_bbox_h // 2)
                    y_max = min(img_h, bbox_center_y + new_bbox_h // 2)
                    x_min = max(0, bbox_center_x - new_bbox_w // 2)
                    x_max = min(img_w, bbox_center_x + new_bbox_w // 2)

                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

        elif self.mode.startswith("high_") or self.mode.startswith("low_"):
            base_image = None

            if "whole" in self.mode:
                base_image = image
            elif "lesion" in self.mode:
                base_image = image * mask[:, :, np.newaxis]
            elif "background" in self.mode:
                base_image = image * (1 - mask[:, :, np.newaxis])

            if base_image is not None:
                if "high_" in self.mode:
                    # processed_image = high_pass_filter(base_image)
                    processed_image = high_pass_filter(base_image, sigma=1, grayscale=True)
                else:
                    processed_image = low_pass_filter(base_image, sigma=1)
                    
        if self.return_pil:
            processed_image = Image.fromarray(processed_image.astype(np.uint8))
        else:
            if self.transform: 
                processed_image = Image.fromarray(processed_image.astype(np.uint8))
                processed_image = self.transform(processed_image)
        label = torch.tensor(label, dtype=torch.long)
        
        return processed_image, label

class PH2Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, mode="whole", return_pil=False, return_mask=False, 
                 augmented=None, clean=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image names and labels.
            image_dir (str): Directory containing original images.
            mask_dir (str): Directory containing ground truth segmentations.
            transform (callable, optional): Optional transform to apply to images.
            mode (str): One of "whole", "lesion", "background", "bbox", "bbox70", 
                        "bbox90", "high_whole", "low_whole", "high_lesion",
                        "low_lesion", "high_background", "low_background".
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.return_pil = return_pil
        self.return_mask = return_mask
        self.augmented = augmented
        self.clean = clean
        self.mapping = {
                        "dark_corner": "/mnt/scratch-lids/scratch/qixuanj/chil2025/img2img/revised_PH2/dark_corner_6/", 
                        "gel_bubble": "/mnt/scratch-lids/scratch/qixuanj/chil2025/img2img/revised_PH2/gel_bubble_1/", 
                        "ink": "/mnt/scratch-lids/scratch/qixuanj/chil2025/img2img/revised_PH2/ink_4_no_ruler/", 
                        "patches": "/mnt/scratch-lids/scratch/qixuanj/chil2025/img2img/revised_PH2/patches_3/", 
                        "ruler": "/mnt/scratch-lids/scratch/qixuanj/chil2025/img2img/revised_PH2/ruler_1/", 
                    }
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        if self.augmented: 
            img_name = self.df.iloc[idx]['Name']
            aug_name = self.df.iloc[idx][f"{self.augmented}_augmented"]
            img_path = os.path.join(self.mapping[self.augmented], f"img_{aug_name}.png")
        elif self.clean: 
            img_name = self.df.iloc[idx]['Name']
            aug_name = self.df.iloc[idx][self.clean]
            img_path = f"/data/healthy-ml/scratch/qixuanj/new_generative_validation/inpaint/PH2/{self.clean}/{aug_name}"
        else: 
            img_name = self.df.iloc[idx]['Name']
            img_path = os.path.join(self.image_dir, f"{img_name}/{img_name}_Dermoscopic_Image/{img_name}.bmp")
            
        label = self.df.iloc[idx]['label']
        
        mask_path = os.path.join(self.image_dir, f"{img_name}/{img_name}_lesion/{img_name}_lesion.bmp")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        if mask.dtype == bool:  # Convert boolean mask to uint8
            mask = mask.astype(np.uint8) * 255
        
        # Ensure images and masks are the same size
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 0).astype(np.uint8)
        
        if self.mode == "whole":
            processed_image = image
        
        elif self.mode == "lesion":
            processed_image = image * mask[:, :, np.newaxis]

        elif self.mode == "background":
            processed_image = image * (1 - mask[:, :, np.newaxis])

        elif self.mode in ["bbox", "bbox70", "bbox90"]:
            # Compute bounding box around lesion
            y_idxs, x_idxs = np.where(mask > 0)
            if len(y_idxs) == 0 or len(x_idxs) == 0:  # If no lesion
                processed_image = image * 0  # Blackout image
            else:
                y_min, y_max = y_idxs.min(), y_idxs.max()
                x_min, x_max = x_idxs.min(), x_idxs.max()
                
                # Compute the original bbox (for `bbox`)
                if self.mode == "bbox":
                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

                # Expand bbox for bbox70 and bbox90
                else:
                    expand_ratio = 0.7 if self.mode == "bbox70" else 0.9

                    img_h, img_w = image.shape[:2]
                    bbox_h = y_max - y_min
                    bbox_w = x_max - x_min

                    # Calculate expansion to reach desired percentage of total image
                    target_area = expand_ratio * img_h * img_w
                    bbox_center_y, bbox_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                    
                    # Compute new bbox size
                    new_bbox_h = int(np.sqrt(target_area * (bbox_h / bbox_w)))  # Keep aspect ratio
                    new_bbox_w = int(np.sqrt(target_area * (bbox_w / bbox_h)))

                    # Ensure it fits within image boundaries
                    y_min = max(0, bbox_center_y - new_bbox_h // 2)
                    y_max = min(img_h, bbox_center_y + new_bbox_h // 2)
                    x_min = max(0, bbox_center_x - new_bbox_w // 2)
                    x_max = min(img_w, bbox_center_x + new_bbox_w // 2)

                    processed_image = image.copy()
                    cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)

        elif self.mode.startswith("high_") or self.mode.startswith("low_"):
            base_image = None

            if "whole" in self.mode:
                base_image = image
            elif "lesion" in self.mode:
                base_image = image * mask[:, :, np.newaxis]
            elif "background" in self.mode:
                base_image = image * (1 - mask[:, :, np.newaxis])

            if base_image is not None:
                if "high_" in self.mode:
                    # processed_image = high_pass_filter(base_image)
                    processed_image = high_pass_filter(base_image, sigma=1, grayscale=True)
                else:
                    processed_image = low_pass_filter(base_image, sigma=1)
                    
        if self.return_pil:
            processed_image = Image.fromarray(processed_image.astype(np.uint8))
        else:
            if self.transform: 
                processed_image = Image.fromarray(processed_image.astype(np.uint8))
                processed_image = self.transform(processed_image)
        label = torch.tensor(label, dtype=torch.long)

        if self.return_mask: 
            return processed_image, label, mask
        else:
            return processed_image, label

class BCN20000Dataset(Dataset):
    def __init__(self, df, image_dir, transform=None, mode="whole", return_pil=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing image names and labels.
            image_dir (str): Directory containing original images.
            mask_dir (str): Directory containing ground truth segmentations.
            transform (callable, optional): Optional transform to apply to images.
            mode (str): One of "whole", "high_whole", "low_whole"
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.return_pil = return_pil
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_name = self.df.iloc[idx]['isic_id']
        label = self.df.iloc[idx]['label']
        
        img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        

        if self.mode == "whole":
            processed_image = image

        elif self.mode.startswith("high_") or self.mode.startswith("low_"):
            base_image = None

            if "whole" in self.mode:
                base_image = image
        
            if base_image is not None:
                if "high_" in self.mode:
                    processed_image = high_pass_filter(base_image, sigma=1, grayscale=True)
                else:
                    processed_image = low_pass_filter(base_image, sigma=1)
                    
        if self.return_pil:
            processed_image = Image.fromarray(processed_image.astype(np.uint8))
        else:
            if self.transform: 
                processed_image = Image.fromarray(processed_image.astype(np.uint8))
                processed_image = self.transform(processed_image)
        label = torch.tensor(label, dtype=torch.long)
        
        return processed_image, label