import os
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import hausdorff_distance
from LOADER_Segmentation_LesionsCancereuses import Polyp_Dataset, load_data_to_model
from SIA_METRICS_Segmentation_LesionsCancereuses import DiceLoss, IoULoss
from FCBSWINV2_TRANSFORMER_Segmentation_LesionsCancereuses import FCBSwinV2_Transformer

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Testing FCB-SwinV2 Transformer for Lung Lesion Segmentation")
parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--save_results', type=str, required=True, help='Path to save test results')
parser.add_argument('--image_size', type=int, default=384, help='Image size for testing')
args = parser.parse_args()

# Load test dataset
test_csv = os.path.join(args.test_path, "test_data.csv")
test_IDs, test_X, test_Y = load_data_to_model(args.image_size, test_csv)

# Create dataset and DataLoader
test_dataset = Polyp_Dataset(test_IDs, test_X, test_Y, geo_transform=None, color_transform=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCBSwinV2_Transformer(size=args.image_size, checkpoint_path="")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Define evaluation metrics
dice_loss = DiceLoss()
iou_loss = IoULoss()

def surface_dice(y_pred, y_true, threshold=0.5):
    """Computes the Surface Dice Score."""
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    intersection = np.logical_and(y_pred_bin, y_true_bin).sum()
    return 2.0 * intersection / (y_pred_bin.sum() + y_true_bin.sum() + 1e-6)

# Save results file
results_file = os.path.join(args.save_results, "test_results.csv")

# Check if file exists, otherwise create it
if not os.path.exists(results_file):
    pd.DataFrame(columns=["Metric", "Value"]).to_csv(results_file, index=False)

# Initialize metrics
dice_scores = []
iou_scores = []
hausdorff_distances = []
surface_dice_scores = []
recalls = []

# Storage for visualization
image_samples = []
mask_samples = []
prediction_samples = []

# Testing loop
with torch.no_grad():
    for i, (_, img_test, mask_test) in enumerate(test_loader):
        img_test, mask_test = img_test.to(device), mask_test.float().to(device)

        # Model prediction
        pred = model(img_test)
        pred = torch.sigmoid(pred)
        pred_np = pred.cpu().numpy().squeeze()
        mask_np = mask_test.cpu().numpy().squeeze()

        # Compute metrics
        dice_scores.append(1 - dice_loss(pred, mask_test).item())
        iou_scores.append(1 - iou_loss(pred, mask_test).item())
        hausdorff_distances.append(hausdorff_distance(pred_np, mask_np))
        surface_dice_scores.append(surface_dice(pred_np, mask_np))
        recalls.append(np.sum(pred_np * mask_np) / (np.sum(mask_np) + 1e-6))

        # Store images for visualization
        if i < 5:  # Store 5 examples
            image_samples.append(img_test.cpu().numpy().squeeze().transpose(1, 2, 0))
            mask_samples.append(mask_np)
            prediction_samples.append(pred_np)

# Save results
results = {
    "Metric": ["Dice", "IoU", "Hausdorff Distance", "Surface Dice", "Recall"],
    "Value": [
        np.mean(dice_scores),
        np.mean(iou_scores),
        np.mean(hausdorff_distances),
        np.mean(surface_dice_scores),
        np.mean(recalls),
    ],
}

df_results = pd.DataFrame(results)
df_results.to_csv(results_file, index=False)

# Visualization of predictions
fig, axs = plt.subplots(5, 3, figsize=(10, 15))

for i in range(5):
    axs[i, 0].imshow(image_samples[i], cmap="gray")
    axs[i, 0].set_title("Original Image")

    axs[i, 1].imshow(mask_samples[i], cmap="gray")
    axs[i, 1].set_title("Ground Truth Mask")

    axs[i, 2].imshow(prediction_samples[i], cmap="gray")
    axs[i, 2].set_title("Model Prediction")

plt.tight_layout()
plt.savefig(os.path.join(args.save_results, "test_predictions.png"))
plt.show()
