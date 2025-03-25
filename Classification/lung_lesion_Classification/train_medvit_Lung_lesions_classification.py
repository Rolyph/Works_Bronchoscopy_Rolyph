import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Early stopping mechanism to prevent overfitting
class EarlyStopping:
    def __init__(self, patience=15, delta=0.001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, score, model, path='checkpoint.pth'):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# MedViT-based classification model for lung lesion detection
class MedViT_Lung_Lesions(nn.Module):
    def __init__(self, num_classes):
        super(MedViT_Lung_Lesions, self).__init__()
        from timm import create_model
        self.backbone = create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.dropout = nn.Dropout(0.4)  # Increased dropout to 40%

    def forward(self, x):
        x = self.backbone(x)
        return self.dropout(x)

# Function to compute classification metrics
def calculate_metrics(predictions, labels):
    preds = torch.sigmoid(predictions).cpu().numpy()
    labels = labels.cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)

    precision = precision_score(labels, preds_binary, average='macro', zero_division=0)
    recall = recall_score(labels, preds_binary, average='macro', zero_division=0)
    f1 = f1_score(labels, preds_binary, average='macro', zero_division=0)
    auc_roc = roc_auc_score(labels, preds, average='macro', multi_class='ovr')
    
    return precision, recall, f1, auc_roc

# Custom dataset class for lung lesion classification
class LungLesionsDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_json_path, transform=None, min_label_occurrences=20):
        self.images_dir = images_dir
        self.transform = transform
        
        # Load label annotations
        with open(labels_json_path, 'r') as f:
            self.labels = json.load(f)

        # Exclude rare labels
        label_counts = Counter([entry['label_name'] for entry in self.labels])
        self.excluded_labels = [label for label, count in label_counts.items() if count < min_label_occurrences]
        self.labels = [entry for entry in self.labels if entry['label_name'] not in self.excluded_labels]

        self.image_filenames = sorted(os.listdir(images_dir))
        self.class_list = [
            'Muscosal erythema', 'Anthrocosis', 'Stenosis', 'Mucosal edema of carina',
            'Mucosal infiltration', 'Vascular growth', 'Tumor'
        ]
        # Remove excluded classes
        self.class_list = [cls for cls in self.class_list if cls not in self.excluded_labels]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Retrieve multi-class labels
        object_id = os.path.splitext(img_name)[0]
        label_tensor = torch.zeros(len(self.class_list))
        for entry in self.labels:
            if entry['object_id'] == object_id:
                if entry['label_name'] in self.class_list:
                    class_idx = self.class_list.index(entry['label_name'])
                    label_tensor[class_idx] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="MedViT Classification Training Script for Lung Lesions")
parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset (images and masks)')
parser.add_argument('--label_json_path', type=str, required=True, help='Path to JSON file containing labels')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size (default: 8)')
parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Reduced learning rate')
parser.add_argument('--tensorboard_log_dir', type=str, default='./tensorboard_logs', help='Directory for TensorBoard logs')
parser.add_argument('--min_label_occurrences', type=int, default=20, help='Exclude labels with fewer occurrences')
args = parser.parse_args()

# Configure TensorBoard
writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

# Define image transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load training and validation datasets
train_dataset = LungLesionsDataset(
    images_dir=os.path.join(args.dataset_path, 'train/imgs'),
    labels_json_path=args.label_json_path,
    transform=transform,
    min_label_occurrences=args.min_label_occurrences
)
val_dataset = LungLesionsDataset(
    images_dir=os.path.join(args.dataset_path, 'val/imgs'),
    labels_json_path=args.label_json_path,
    transform=transform,
    min_label_occurrences=args.min_label_occurrences
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedViT_Lung_Lesions(num_classes=len(train_dataset.class_list)).to(device)

# Optimizer, scheduler, and loss function
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
criterion = nn.BCEWithLogitsLoss()
early_stopping = EarlyStopping(patience=15, verbose=True)

# Training loop
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_preds.append(outputs)
        all_labels.append(labels)

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

    val_loss /= len(val_loader)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision, recall, f1, auc_roc = calculate_metrics(all_preds, all_labels)

    print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

    early_stopping(f1, model, path="best_medvit_lung_lesions.pth")
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

scheduler.step(val_loss)
writer.close()
