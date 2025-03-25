import os
import torch
import json
import argparse
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
from PIL import Image
import imageio
from skimage import img_as_ubyte
import torchvision.transforms as transforms

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Dataset Splitting for ESFPNet-based model")
parser.add_argument('--label_json_path', type=str, required=True, help='Path to the JSON file containing labels')
parser.add_argument('--path_cancer_imgs', type=str, required=True, help='Path to cancer case images')
parser.add_argument('--path_non_cancer_imgs', type=str, required=True, help='Path to non-cancer case images')
parser.add_argument('--path_cancer_masks', type=str, required=True, help='Path to cancer case masks')
parser.add_argument('--path_non_cancer_masks', type=str, required=True, help='Path to non-cancer case masks')
parser.add_argument('--path_dataset', type=str, required=True, help='Path to save the dataset')
parser.add_argument('--task', type=str, required=True, choices=['Anatomical_landmarks', 'Lung_lesions'], 
                    help='Task: Anatomical_landmarks or Lung_lesions')
args = parser.parse_args()

# Clear CUDA cache
torch.cuda.empty_cache()


def collect_image_paths(root_dir):
    """
    Collects all image paths from the given directory structure.
    Assumes each patient has a single subfolder containing images.
    """
    image_paths = []
    for patient_folder in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_folder)
        if os.path.isdir(patient_path):
            subfolders = os.listdir(patient_path)
            if len(subfolders) == 1:  # Assumes a single subfolder per patient
                image_subfolder = os.path.join(patient_path, subfolders[0])
                if os.path.isdir(image_subfolder):
                    for file in os.listdir(image_subfolder):
                        if file.endswith('.jpg') or file.endswith('.png'):
                            image_paths.append(os.path.join(image_subfolder, file))
    return image_paths


class SplittingDataset(Dataset):
    """
    Dataset loader for segmentation tasks.
    """
    def __init__(self, image_root, gt_root):
        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        object_ids = {entry['object_id'] for entry in data}

        self.images = collect_image_paths(image_root)
        self.gts = collect_image_paths(gt_root)

        # Filter images and masks that have corresponding labels
        self.images = [img for img in self.images if os.path.splitext(os.path.basename(img))[0] in object_ids]
        self.gts = [gt for gt in self.gts if os.path.splitext(os.path.basename(gt))[0] in object_ids]

        # Sort to ensure correct mapping
        self.images, self.gts = zip(*sorted(zip(self.images, self.gts), key=lambda x: x[0]))

        self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        name_image = os.path.basename(self.images[index])

        file_name = os.path.splitext(name_image)[0]

        with open(args.label_json_path, 'r') as f:
            data = json.load(f)

        if args.task == 'Anatomical_landmarks':
            label_list = [
                'Vocal cords', 'Main carina', 'Intermediate bronchus', 
                'Right superior lobar bronchus', 'Right inferior lobar bronchus', 
                'Right middle lobar bronchus', 'Left inferior lobar bronchus', 
                'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea'
            ]
        else:
            label_list = [
                'Muscosal erythema', 'Anthrocosis', 'Stenosis', 
                'Mucosal edema of carina', 'Mucosal infiltration', 
                'Vascular growth', 'Tumor'
            ]

        label_name = [entry['label_name'] for entry in data if entry['object_id'] == file_name]
        label_tensor = torch.zeros(len(label_list))
        for name in label_name:
            if name in label_list:
                label_tensor[label_list.index(name)] = 1

        return self.transform(image), self.transform(gt), label_tensor, name_image

    def filter_files(self):
        """ Ensure images and masks have the same size and filter out mismatched pairs. """
        assert len(self.images) == len(self.gts), "Mismatch between images and masks count."
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        """ Load an RGB image. """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        """ Load a binary mask image. """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def split_dataset():
    """ Splits the dataset into training, validation, and test sets and saves them to the specified directory. """
    # Create directories for splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.path_dataset, split, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(args.path_dataset, split, 'masks'), exist_ok=True)

    # Create datasets for cancer and non-cancer cases
    dataset_list = [
        SplittingDataset(args.path_cancer_imgs, args.path_cancer_masks),
        SplittingDataset(args.path_non_cancer_imgs, args.path_non_cancer_masks)
    ]

    whole_dataset = ConcatDataset(dataset_list)

    # Extract images, masks, labels, and filenames
    imgs_list, masks_list, labels_list, names_list = zip(*list(whole_dataset))

    # Ensure a minimum occurrence of labels before inclusion
    X_data, Y_data = [], []
    for img, mask, lbl, name in zip(imgs_list, masks_list, labels_list, names_list):
        if labels_list.count(lbl) >= 5:
            X_data.append((img, mask, name, lbl))
            Y_data.append(lbl)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=42, stratify=Y_data)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # Save splits
    for data, img_path, mask_path in zip(
        [X_train, X_val, X_test], 
        [os.path.join(args.path_dataset, 'train', 'imgs'), os.path.join(args.path_dataset, 'val', 'imgs'), os.path.join(args.path_dataset, 'test', 'imgs')],
        [os.path.join(args.path_dataset, 'train', 'masks'), os.path.join(args.path_dataset, 'val', 'masks'), os.path.join(args.path_dataset, 'test', 'masks')]
    ):
        for img, mask, name, lbl in data:
            imageio.imwrite(os.path.join(img_path, name), img_as_ubyte(img.data.cpu().numpy().transpose(1, 2, 0).squeeze()))
            imageio.imwrite(os.path.join(mask_path, name), img_as_ubyte(mask.data.cpu().numpy().squeeze()))


split_dataset()
