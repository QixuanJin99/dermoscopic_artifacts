# Standard library imports
import os
import sys
import json
import pickle
import random
import re
from glob import glob
from pathlib import Path
import gc

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
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers import UNet2DConditionModel
from peft import PeftModel, LoraConfig
import argparse
from concurrent.futures import ThreadPoolExecutor
from datasets import ISICDataset, HAM10000Dataset, PH2Dataset, BCN20000Dataset

parser = argparse.ArgumentParser(description='Generate ISIC augmented images.')
parser.add_argument('--dataset', type=str, default="ISIC")
parser.add_argument('--ckpt_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--keyword', type=str, default="malignant")
args = parser.parse_args()


def get_lora_sd_inpaint_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default",
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")

    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")
    
    # Load UNet for inpainting
    unet = UNet2DConditionModel.from_pretrained(f"/mnt/scratch-lids/scratch/qixuanj/isic_sd_base/checkpoint-2000/unet")
    
    # Load Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model_name_or_path, unet=unet, torch_dtype=dtype, safety_checker=None
    ).to(device)
    
    # Load LoRA adapters for UNet and text encoder
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

# Function to save images in parallel
def save_image(image, filename):
    image.save(filename)

token_mapping = {
"patches": "olis",
"dark_corner": "lun", 
"ruler": "dits", 
"ink": "httr", 
"gel_border": "rcn", 
"gel_bubble": "sown", 
"benign": "waj", 
"malignant": "shld", 
"no_artifact": "adl", 
}

strength_mapping = {
"patches": 0.85, 
"dark_corner": 0.75, 
"ruler": 0.65, 
"ink": 0.7, 
"gel_bubble": 0.6, 
}


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

dataset_mode = "whole"

if args.dataset == "ISIC": 
    df = pd.read_csv("/data/healthy-ml/scratch/qixuanj/debiasing-skin/artefacts-annotation/isic_bias.csv", index_col=0)
    full_dataset = ISICDataset(df, image_dir, mask_dir, mode=dataset_mode, return_pil=True)
elif args.dataset == "HAM10000":
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/HAM10000_metadata_with_objects.csv", index_col=0)
    full_dataset = HAM10000Dataset(df, image_dir, mask_dir, mode=dataset_mode, return_pil=True)
elif args.dataset == "PH2": 
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/ph2_metadata.csv", index_col=0)
    full_dataset = PH2Dataset(df, image_dir, mode=dataset_mode, return_pil=True, return_mask=True)
elif args.dataset == "BCN20000": 
    df = pd.read_csv("/mnt/scratch-lids/scratch/qixuanj/chil2025/metadata/bcn20000_metadata.csv", index_col=0)
    full_dataset = BCN20000Dataset(df, image_dir, mode=dataset_mode, return_pil=True)
    

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True) 

pipe = get_lora_sd_inpaint_pipeline(
    ckpt_dir = Path(args.ckpt_dir),
    base_model_name_or_path="runwayml/stable-diffusion-v1-5", 
    adapter_name=args.keyword,
)

batch_size = 5  # Num duplicates per image 
guidance_scale = 10  # Reduce if speed is more important than fidelity

# Optimize inference
pipe.enable_attention_slicing()  # Reduce memory usage
pipe.enable_sequential_cpu_offload()  # Offload computation when running many images

keyword = args.keyword.replace("_", " ")
prompt = f"a dermoscopic image of {token_mapping[args.keyword]} {keyword}"

num_images = int(len(full_dataset) * batch_size)

# Generate and save images in batches
with ThreadPoolExecutor() as executor:  # Use parallel saving
    for batch_start in range(0, num_images, batch_size):
        
        j = int(batch_start//batch_size)
        source_image = full_dataset[j][0].resize((224, 224), Image.BILINEAR)
        binary_mask = full_dataset[j][2]
        mask_pil = Image.fromarray((1 - binary_mask) * 255).convert("L").resize((224, 224), Image.BILINEAR)
 
        images = pipe(prompt=prompt,
                        image=source_image,
                        mask_image=mask_pil,
                        strength=strength_mapping[args.keyword], 
                        guidance=10, 
                        num_inference_steps=20,
                        num_images_per_prompt=batch_size).images 
    
            
        # Save images in parallel
        original_index = int(batch_start // batch_size)
        futures = []
        for i, img in enumerate(images):
            img_path = os.path.join(save_dir, f"img_{original_index}_{i}.png")
            futures.append(executor.submit(save_image, img, img_path))
        
        # Wait for all images in the batch to be saved
        for future in futures:
            future.result()
            
        # Free up memory after processing each batch
        del images
        torch.cuda.empty_cache()  # Clear CUDA memory
        gc.collect()  # Run garbage collection to free unused memory

print(f"Images saved to {save_dir}")