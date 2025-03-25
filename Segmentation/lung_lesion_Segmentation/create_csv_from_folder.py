import os
import csv
import argparse
from PIL import Image

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Generate CSV files from dataset folder structure")
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset root directory')
args = parser.parse_args()

# Function to validate images and masks using PIL
def validate_images_and_masks(folder_path):
    """
    Checks the integrity of image and mask files in the given directory.
    """
    print(f"Validating images and masks in folder: {folder_path}")

    for file_name in os.listdir(folder_path):
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Validate image
                img = Image.open(file_path)
                img.verify()  # Quick integrity check
            except Exception as e:
                print(f"Warning: Invalid image detected ({file_name}) - {e}")
                continue

# Function to create a CSV file from a folder containing images and masks
def create_csv_from_folder(folder_path, split_name):
    """
    Generates a CSV file listing images and corresponding masks.
    """
    print(f"Generating CSV file for folder {folder_path}...")

    # Define output CSV file path
    csv_file = os.path.join(folder_path, f"{split_name}_data.csv")

    # Create and write to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "mask_path"])  # Header row

        # Get image and mask directories
        image_dir = os.path.join(folder_path, "imgs")
        mask_dir = os.path.join(folder_path, "masks")

        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            print(f"Error: Missing 'imgs' or 'masks' directory in {folder_path}. Skipping CSV generation.")
            return

        for image_file in sorted(os.listdir(image_dir)):
            if image_file.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(image_dir, image_file)
                mask_path = os.path.join(mask_dir, image_file)

                # Ensure corresponding mask exists
                if os.path.exists(mask_path):
                    writer.writerow([image_path, mask_path])
                else:
                    print(f"Warning: Missing mask for image: {image_file}")

    print(f"CSV file successfully created: {csv_file}")

# Paths to dataset splits
train_path = os.path.join(args.dataset_path, 'train')
val_path = os.path.join(args.dataset_path, 'val')
test_path = os.path.join(args.dataset_path, 'test')

# Validate and create CSV files for each dataset split
for split, path in zip(['train', 'val', 'test'], [train_path, val_path, test_path]):
    print(f"Processing folder {split}: {path}")

    # Validate image/mask files
    if os.path.exists(path):
        validate_images_and_masks(path)

        # Generate CSV file
        try:
            create_csv_from_folder(path, split)
        except Exception as e:
            print(f"Error generating CSV for {split}: {e}")
    else:
        print(f"Warning: {split} directory does not exist, skipping.")

print("CSV file generation completed successfully!")
