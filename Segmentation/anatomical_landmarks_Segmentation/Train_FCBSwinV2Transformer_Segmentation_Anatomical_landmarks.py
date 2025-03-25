import os
import time
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import albumentations as A
import LOADER_Segmentation_Anatomical_landmarks as LOADER
import SIA_METRICS_Segmentation_Anatomical_landmarks as SIA_METRICS
import FCBSWINV2_TRANSFORMER_Segmentation_Anatomical_landmarks as FCBSWINV2_TRANSFORMER

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="Training script for FCB-SwinV2 Transformer on Anatomical Landmarks")
parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--val_path', type=str, required=True, help='Path to the validation dataset')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model and results')
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs (default: 150)')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training (default: 2)')
parser.add_argument('--val_batch_size', type=int, default=1, help='Batch size for validation (default: 1)')
parser.add_argument('--image_size', type=int, default=384, help='Image size for training (default: 384)')
args = parser.parse_args()

# Paths to training and validation CSV files
train_csv = os.path.join(args.train_path, "train_data.csv")
val_csv = os.path.join(args.val_path, "val_data.csv")

# Load dataset from CSV files
train_IDs, train_X, train_Y = LOADER.load_data_to_model(args.image_size, train_csv)
val_IDs, val_X, val_Y = LOADER.load_data_to_model(args.image_size, val_csv)

# Define data augmentations
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
train_dataset = LOADER.AnatomicalLandmarksDataset(train_IDs, train_X, train_Y, geo_transform=geometric, color_transform=color)
val_dataset = LOADER.AnatomicalLandmarksDataset(val_IDs, val_X, val_Y, geo_transform=None, color_transform=None)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=4)

# Initialize model
model = FCBSWINV2_TRANSFORMER.FCBSwinV2_Transformer(size=args.image_size, checkpoint_path="")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer, scheduler, and loss functions
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10)
criterion = SIA_METRICS.DiceBCELoss()
dice = SIA_METRICS.DiceLoss()

# Variables for storing training results
save_string = os.path.join(args.save_path, "1st_training_SGM_Anatomical")
epochs = args.epochs

train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

# Model training loop
for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_dice = 0
    train_batches = 0

    start_time = time.time()
    for _, img_train, mask_train in train_loader:
        img_train, mask_train = img_train.to(device), mask_train.float().to(device)

        # Forward pass and loss computation
        pred = model(img_train)
        pred = torch.sigmoid(pred)
        loss = criterion(pred, mask_train)
        dice_score = 1 - dice(pred, mask_train)

        # Optimization
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

    # Adjust learning rate
    scheduler.step(avg_val_loss)

    # Print training progress
    print(f"Epoch {epoch+1}/{epochs}:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")
    print(f"Time: {time.time() - start_time:.2f}s")

    # Save the model if the performance improves
    if avg_val_dice > max(val_dice_scores[:-1] or [0]):
        torch.save(model.state_dict(), save_string + "_best_model.pt")
        print("Model saved!")

# Save training results to CSV
results = pd.DataFrame({
    "Epoch": list(range(1, epochs + 1)),
    "Train Loss": train_losses,
    "Val Loss": val_losses,
    "Train Dice": train_dice_scores,
    "Val Dice": val_dice_scores
})
results.to_csv(save_string + "_training_results.csv", index=False)
print("Training complete, results saved.")
