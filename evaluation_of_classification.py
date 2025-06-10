import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset
from road_sign_detection.data.annotations import check_annotations

# ------------------- Configuration -------------------

# Dataset and model paths
root: str = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
check_annotations(root)
annotations: str = os.path.join(root, 'test_annotations.json')
model_path: str = os.path.join(os.getcwd(), 'models', 'classification_model_2_fold2.pth')  # Final averaged model

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

# ------------------- Dataset -------------------

# Load test dataset
test_dataset = TT100KSignDataset(annotations, root, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------- Model Setup -------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
model = CNNClassifier(len(test_dataset.annotations['types'])).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# ------------------- Evaluation -------------------

running_loss = 0.0
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ------------------- Results -------------------

avg_loss = running_loss / len(test_loader)
accuracy = correct / total

print(f"\nTest Loss       : {avg_loss:.4f}")
print(f"Test Accuracy   : {accuracy:.4f}")

# Optional: print baseline accuracy with random guessing
random_guess_acc = 1 / len(test_dataset.annotations['types'])
print(f"Random Guess Accuracy: {random_guess_acc:.4f}")
if accuracy > random_guess_acc:
    print("=> Model outperforms random guessing.")
else:
    print("=> Model does not outperform random guessing.")
