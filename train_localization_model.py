"""
Training script for bounding box regression on TT100K dataset.
Dataset must be under:
Applied-Machine-Learning/project_name/data/tt100k_2021
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from project_name.models.localization_base_model import CNNDetector
from project_name.data.dataset_loader import TT100KDataset

# Paths
root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + 'train/ids.txt'
model_path = os.getcwd() + '/models/cnn_detector.pth'

# Filter annotations for training IDs
if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    filtered_imgs = {
        img_id: img_data for img_id, img_data in annos['imgs'].items() if img_id in ids
    }

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

# Transforms (64x64 grayscale, normalized)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# Load dataset
tt100k_data = TT100KDataset(filtered_annotations, root, transform)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size
train_split, val_split = random_split(tt100k_data, [t_size, v_size])
t_loader = DataLoader(train_split, batch_size=32, shuffle=True)
v_loader = DataLoader(val_split, batch_size=32, shuffle=True)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
model = CNNDetector().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 10

# Training loop
for epoch in range(epochs):
    print(f'\nEpoch [{epoch + 1}/{epochs}]')
    model.train()
    total_train_loss = 0.0

    for images, targets in tqdm(t_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(t_loader)
    print(f'Average training loss: {avg_train_loss:.4f}')

    # Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, targets in v_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(v_loader)
    print(f'Average validation loss: {avg_val_loss:.4f}')

# Save model
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
