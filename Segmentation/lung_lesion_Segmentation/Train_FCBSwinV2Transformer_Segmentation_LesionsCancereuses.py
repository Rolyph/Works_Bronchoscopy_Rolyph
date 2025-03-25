import os
import time
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
import LOADER_Segmentation_LesionsCancereuses as LOADER
import SIA_METRICS_Segmentation_LesionsCancereuses as SIA_METRICS
import FCBSWINV2_TRANSFORMER_Segmentation_LesionsCancereuses as FCBSWINV2_TRANSFORMER

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Training FCB-SwinV2 Transformer for Lung Lesion Segmentation")
parser.add_argument('--train_path', type=str, required=True, help='Path to training dataset')
parser.add_argument('--val_path', type=str, required=True, help='Path to validation dataset')
parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
parser.add_argument('--save_path', type=str, required=True, help='Path to save training results')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
parser.add_argument('--image_size', type=int, default=384, help='Image size for training')
args = parser.parse_args()

# Load dataset CSV files
train_csv = os.path.join(args.train_path, "train_data.csv")
val_csv = os.path.join(args.val_path, "val_data.csv")
test_csv = os.path.join(args.test_path, "test_data.csv")

train_IDs, train_X, train_Y = LOADER.load_data_to_model(args.image_size, train_csv)
val_IDs, val_X, val_Y = LOADER.load_data_to_model(args.image_size, val_csv)
test_IDs, test_X, test_Y = LOADER.load_data_to_model(args.image_size, test_csv)

# Check min/max values in masks before training
print(f"Train mask - min: {train_Y.min()}, max: {train_Y.max()}")
print(f"Val mask - min: {val_Y.min()}, max: {val_Y.max()}")

# Define augmentations
geometric = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Transpose(p=0.5),
    A.Affine(scale=(0.5, 1.5), translate_percent=(-0.125, 0.125), rotate=(-180, 180), shear=(-22.5, 22), always_apply=True)
])

color = A.Compose([
    A.ColorJitter(brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01, always_apply=True)
])

# Create datasets
train_dataset = LOADER.Polyp_Dataset(train_IDs, train_X, train_Y, geo_transform=geometric, color_transform=color)
val_dataset = LOADER.Polyp_Dataset(val_IDs, val_X, val_Y, geo_transform=None, color_transform=None)
test_dataset = LOADER.Polyp_Dataset(test_IDs, test_X, test_Y, geo_transform=None, color_transform=None)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Initialize model without checkpoint
model = FCBSWINV2_TRANSFORMER.FCBSwinV2_Transformer(size=args.image_size, checkpoint_path="")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define optimizer, scheduler, and loss functions
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10)
criterion = SIA_METRICS.DiceBCELoss()
dice = SIA_METRICS.DiceLoss()

# Variables for saving results
save_string = os.path.join(args.save_path, "1st_training_SGM_Lung_Lesions")
epochs = args.epochs

train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_dice = 0
    train_batches = 0

    start_time = time.time()
    for _, img_train, mask_train in train_loader:
        img_train, mask_train = img_train.to(device), mask_train.float().to(device)

        # Binarize masks
        mask_train = (mask_train > 0.5).float()

        # Ensure masks have the correct shape
        if mask_train.shape[1] != 1:
            mask_train = mask_train.unsqueeze(1)  # Ensure (B, 1, H, W)

        # Model predictions and loss computation
        pred = model(img_train)
        pred = torch.sigmoid(pred)
        loss = criterion(pred, mask_train)
        dice_score = 1 - dice(pred, mask_train)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        epoch_train_dice += dice_score.item()
        train_batches += 1

    avg_train_loss = epoch_train_loss / train_batches
    avg_train_dice = epoch_train_dice / train_batches
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)

    # Validation
    model.eval()
    epoch_val_loss = 0
    epoch_val_dice = 0
    val_batches = 0

    with torch.no_grad():
        for _, img_val, mask_val in val_loader:
            img_val, mask_val = img_val.to(device), mask_val.float().to(device)

            mask_val = (mask_val > 0.5).float()
            if mask_val.shape[1] != 1:
                mask_val = mask_val.unsqueeze(1)

            pred = model(img_val)
            pred = torch.sigmoid(pred)
            loss = criterion(pred, mask_val)
            dice_score = 1 - dice(pred, mask_val)

            epoch_val_loss += loss.item()
            epoch_val_dice += dice_score.item()
            val_batches += 1

    avg_val_loss = epoch_val_loss / val_batches
    avg_val_dice = epoch_val_dice / val_batches
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    # Save best model
    if avg_val_dice >= max(val_dice_scores[:-1] or [0]):
        torch.save(model.state_dict(), save_string + "_best_model.pt")
        print("Model saved!")

# Save training results
results = pd.DataFrame({
    "Epoch": list(range(1, epochs + 1)),
    "Train Loss": train_losses,
    "Val Loss": val_losses,
    "Train Dice": train_dice_scores,
    "Val Dice": val_dice_scores
})
results.to_csv(save_string + "_training_results.csv", index=False)
print("Training complete and results saved.")
