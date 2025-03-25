import os
import glob
import pandas as pd
import torch
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Loader for Anatomical Landmarks Segmentation")
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset root directory')
parser.add_argument('--csv_name', type=str, required=True, help='Name for the generated CSV file')
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing image and mask paths')
parser.add_argument('--img_size', type=int, required=True, help='Image resizing size')
args = parser.parse_args()


def create_csv_from_folder(folder_path, csv_name):
    """
    Creates a CSV file listing image and mask paths.

    Args:
        folder_path (str): Path to the folder containing 'imgs/' and 'masks/' subdirectories.
        csv_name (str): Name of the generated CSV file.
    """
    images_folder = os.path.join(folder_path, 'imgs/')
    masks_folder = os.path.join(folder_path, 'masks/')
    
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Image folder not found: {images_folder}")
    if not os.path.exists(masks_folder):
        raise FileNotFoundError(f"Mask folder not found: {masks_folder}")
    
    all_images = sorted(glob.glob(os.path.join(images_folder, "*.png")))
    all_masks = sorted(glob.glob(os.path.join(masks_folder, "*.png")))

    if not all_images:
        raise FileNotFoundError(f"No images found in: {images_folder}")
    if not all_masks:
        raise FileNotFoundError(f"No masks found in: {masks_folder}")

    data = {
        'image_path': all_images,
        'mask_path': all_masks
    }
    df = pd.DataFrame(data)
    output_csv_path = os.path.join(folder_path, f"{csv_name}.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file successfully created: {output_csv_path}")


def load_data_to_model(img_size, csv_path):
    """
    Loads data from a CSV file containing image and mask paths.

    Args:
        img_size (int): Image resizing size.
        csv_path (str): Path to the CSV file.

    Returns:
        IDs (list): List of image names.
        X (numpy array): Normalized images.
        Y (numpy array): Binary masks.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    all_images = df['image_path'].tolist()
    all_masks = df['mask_path'].tolist()

    X = np.zeros((len(all_images), img_size, img_size, 3), dtype=np.float32)
    Y = np.zeros((len(all_images), img_size, img_size), dtype=np.uint8)

    IDs = []

    for n, (image_path, mask_path) in tqdm(enumerate(zip(all_images, all_masks)), 
                                           total=len(all_images), 
                                           desc="Loading Data"):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        try:
            image = np.array(Image.open(image_path).resize((img_size, img_size)))
            mask = np.array(Image.open(mask_path).resize((img_size, img_size)))
        except Exception as e:
            raise RuntimeError(f"Error loading {image_path}: {e}")

        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)  # Correct binarization

        X[n] = image / 255.0  # Normalize between 0 and 1
        Y[n] = mask  # Binary masks

        IDs.append(os.path.basename(image_path))

    Y = np.expand_dims(Y, axis=-1)  # Add a channel dimension for PyTorch
    return IDs, X, Y


class AnatomicalLandmarksDataset(Dataset):
    """
    Custom dataset for anatomical landmark segmentation.
    """
    def __init__(self, IDs, X, Y, geo_transform=None, color_transform=None):
        """
        Args:
            IDs (list): List of image names.
            X (numpy array): Images.
            Y (numpy array): Masks.
            geo_transform (callable, optional): Geometric transformations.
            color_transform (callable, optional): Color transformations.
        """
        self.IDs = IDs
        self.X = X
        self.Y = Y
        self.geo_transform = geo_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index]
        mask = self.Y[index]

        if self.geo_transform:
            augmented = self.geo_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self.color_transform:
            augmented = self.color_transform(image=image)
            image = augmented['image']

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)

        return self.IDs[index], image, mask
