import os
import torch
import json
import argparse
import random
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import imageio
from skimage import img_as_ubyte
import torchvision.transforms as transforms

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Strict Patient-Wise Dataset Splitting for ESFPNet-based model")
parser.add_argument('--label_json_path', type=str, required=True, help='Path to the JSON file containing labels')
parser.add_argument('--path_cancer_imgs', type=str, required=True, help='Path to cancer case images')
parser.add_argument('--path_non_cancer_imgs', type=str, required=True, help='Path to non-cancer case images')
parser.add_argument('--path_cancer_masks', type=str, required=True, help='Path to cancer case masks')
parser.add_argument('--path_non_cancer_masks', type=str, required=True, help='Path to non-cancer case masks')
parser.add_argument('--path_dataset', type=str, required=True, help='Path to save the dataset')
parser.add_argument('--task', type=str, required=True, choices=['Anatomical_landmarks', 'Lung_lesions'],
                    help='Task: Anatomical_landmarks or Lung_lesions')
args = parser.parse_args()

def collect_patient_data(root_dir):
    """ Collect patient folders and corresponding images """
    patient_data = {}
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if os.path.isdir(patient_path):
            image_paths = []
            for subfolder in os.listdir(patient_path):
                image_subfolder = os.path.join(patient_path, subfolder)
                if os.path.isdir(image_subfolder):
                    for file in os.listdir(image_subfolder):
                        if file.endswith('.jpg') or file.endswith('.png'):
                            image_paths.append(os.path.join(image_subfolder, file))
            if image_paths:
                patient_data[patient_folder] = image_paths
    return patient_data

def split_by_patient(patient_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """ Strict patient-wise split into train, validation, and test sets """
    patients = list(patient_data.keys())
    random.shuffle(patients)  # Shuffle patients randomly
    
    train_size = int(len(patients) * train_ratio)
    val_size = int(len(patients) * val_ratio)
    
    train_patients = patients[:train_size]
    val_patients = patients[train_size:train_size + val_size]
    test_patients = patients[train_size + val_size:]
    
    return train_patients, val_patients, test_patients

def copy_data(patient_list, patient_data, save_path, mask_type):
    """ Copy images and masks to respective directories """
    img_save_path = os.path.join(save_path, 'imgs')
    mask_save_path = os.path.join(save_path, 'masks')
    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)

    for patient in patient_list:
        for img_path in patient_data[patient]:
            img_name = os.path.basename(img_path)
            
            # Determine the corresponding mask directory based on task type
            if mask_type == "Anatomical_landmarks":
                mask_path = img_path.replace('imgs', 'masks_Anatomical_Landmarks')
            elif mask_type == "Lung_lesions":
                mask_path = img_path.replace('imgs', 'masks_Lung_cancer_lesions')
            else:
                print(f"Unknown task: {mask_type}")
                continue

            # Check if the mask exists and is not empty
            if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                try:
                    image = Image.open(img_path)
                    mask = Image.open(mask_path)

                    # Save images and masks
                    imageio.imwrite(os.path.join(img_save_path, img_name), img_as_ubyte(image))
                    imageio.imwrite(os.path.join(mask_save_path, img_name), img_as_ubyte(mask))
                except Exception as e:
                    print(f"Error reading {img_path} or {mask_path}: {e}")
            else:
                print(f"Skipping empty or missing mask: {mask_path}")

# Load and organize patients
cancer_patients = collect_patient_data(args.path_cancer_imgs)
non_cancer_patients = collect_patient_data(args.path_non_cancer_imgs)

# Ensure data was found
if not cancer_patients and not non_cancer_patients:
    print("No data found! Please check the provided paths.")
    exit(1)

# Strict patient-wise split
train_cancer, val_cancer, test_cancer = split_by_patient(cancer_patients)
train_non_cancer, val_non_cancer, test_non_cancer = split_by_patient(non_cancer_patients)

# Create train/val/test datasets
copy_data(train_cancer, cancer_patients, os.path.join(args.path_dataset, 'train'), args.task)
copy_data(val_cancer, cancer_patients, os.path.join(args.path_dataset, 'val'), args.task)
copy_data(test_cancer, cancer_patients, os.path.join(args.path_dataset, 'test'), args.task)

copy_data(train_non_cancer, non_cancer_patients, os.path.join(args.path_dataset, 'train'), args.task)
copy_data(val_non_cancer, non_cancer_patients, os.path.join(args.path_dataset, 'val'), args.task)
copy_data(test_non_cancer, non_cancer_patients, os.path.join(args.path_dataset, 'test'), args.task)

print("Strict patient-wise dataset split completed successfully!")
