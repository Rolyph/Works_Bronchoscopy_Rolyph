import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# MedViT model for anatomical landmark classification
class MedViT_Anatomical_Landmarks(nn.Module):
    def __init__(self, num_classes):
        super(MedViT_Anatomical_Landmarks, self).__init__()
        from timm import create_model
        self.backbone = create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.backbone(x)
        return self.dropout(x)

# Custom dataset for anatomical landmark classification
class AnatomicalLandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_json_path, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        # Load label annotations
        with open(labels_json_path, 'r') as f:
            self.labels = json.load(f)

        self.image_filenames = sorted(os.listdir(images_dir))
        self.class_list = [
            'Vocal cords', 'Main carina', 'Intermediate bronchus', 'Right superior lobar bronchus',
            'Right inferior lobar bronchus', 'Right middle lobar bronchus', 'Left inferior lobar bronchus',
            'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea'
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        object_id = os.path.splitext(img_name)[0]
        label_tensor = torch.zeros(len(self.class_list))
        for entry in self.labels:
            if entry['object_id'] == object_id:
                class_idx = self.class_list.index(entry['label_name'])
                label_tensor[class_idx] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Function to compute evaluation metrics
def calculate_metrics(predictions, labels):
    preds = torch.sigmoid(predictions).cpu().numpy()
    labels = labels.cpu().numpy()
    preds_binary = (preds > 0.5).astype(int)

    precision = precision_score(labels, preds_binary, average='macro', zero_division=0)
    recall = recall_score(labels, preds_binary, average='macro', zero_division=0)
    f1 = f1_score(labels, preds_binary, average='macro', zero_division=0)
    auc_roc = roc_auc_score(labels, preds, average='macro', multi_class='ovr')
    accuracy = accuracy_score(labels, preds_binary)

    # Mean Accuracy (MA): Per-class accuracy
    per_class_accuracy = np.mean([accuracy_score(labels[:, i], preds_binary[:, i]) for i in range(labels.shape[1])])
    
    return precision, recall, f1, auc_roc, accuracy, per_class_accuracy

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_list, output_path):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_list, yticklabels=class_list)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_path)
    plt.show()

# Argument parser for command-line execution
parser = argparse.ArgumentParser(description="MedViT Anatomical Landmarks Testing Script")
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--label_json_path', type=str, required=True, help='Path to the label JSON file')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--output_results', type=str, default='results_anatomical.json', help='Output file for test results')
parser.add_argument('--output_conf_matrix', type=str, default='confusion_matrix.png', help='Output file for confusion matrix')
args = parser.parse_args()

# Device selection (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
test_dataset = AnatomicalLandmarksDataset(
    images_dir=os.path.join(args.dataset_path, 'test/imgs'),
    labels_json_path=args.label_json_path,
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Load the trained MedViT model
model = MedViT_Anatomical_Landmarks(num_classes=len(test_dataset.class_list)).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Model testing loop
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        all_preds.append(outputs)
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# Compute evaluation metrics
precision, recall, f1, auc_roc, accuracy, mean_accuracy = calculate_metrics(all_preds, all_labels)

# Display results
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}, Accuracy: {accuracy:.4f}, Mean Accuracy (MA): {mean_accuracy:.4f}")

# Save results to a JSON file
results = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "auc_roc": auc_roc,
    "accuracy": accuracy,
    "mean_accuracy": mean_accuracy
}

with open(args.output_results, 'w') as f:
    json.dump(results, f, indent=4)

# Generate confusion matrix
plot_confusion_matrix(all_labels.cpu().numpy(), torch.sigmoid(all_preds).cpu().numpy(), test_dataset.class_list, args.output_conf_matrix)
