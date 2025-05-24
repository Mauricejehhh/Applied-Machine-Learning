import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from .project_name.models.localization_base_model import CNNDetector
from .project_name.data.dataset_loader import TT100KDataset
from torch.utils.data import random_split, DataLoader
import torchvision.transforms.functional as TF
import cv2
import matplotlib.patches as patches


def denormalize(tensor):
    return tensor * 0.5 + 0.5  # undo normalization to [0, 1] range


def draw_bounding_boxes(img_tensor, bbox, pred_bbox=None):
    """
    img_tensor: [C, H, W] torch.Tensor
    bbox: [x_min, y_min, x_max, y_max] (ground truth)
    pred_bbox: [x_min, y_min, x_max, y_max] (optional predicted)
    """

    img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()  # [H, W, C]
    fig, ax = plt.subplots(1)
    ax.imshow(img_np)

    # Ground truth bbox
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='g',
        facecolor='none',
        label='Ground Truth'
    )
    ax.add_patch(rect)

    # Predicted bbox
    if pred_bbox is not None:
        rect = patches.Rectangle(
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2] - pred_bbox[0],
            pred_bbox[3] - pred_bbox[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none',
            label='Prediction'
        )
        ax.add_patch(rect)

    ax.legend()
    plt.axis('off')
    plt.show()


root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + 'train/ids.txt'
model_path = os.getcwd() + '/models/localization_model.pth'

if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    filtered_imgs = {img_id: img_data
                     for img_id, img_data in annos['imgs'].items()
                     if img_id in ids}

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

tt100k_data = TT100KDataset(filtered_annotations, root, transform)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size

train_split, val_split = random_split(tt100k_data, [t_size, v_size])
t_loader = DataLoader(train_split, 32, shuffle=True)
v_loader = DataLoader(val_split, 32, shuffle=False)

epochs = 5
lr = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')

model = CNNDetector().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    print(f'\nEpoch [{epoch + 1}/{epochs}]')
    model.train()
    running_tloss = 0.0

    for i, data in enumerate(tqdm(t_loader)):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        t_loss = loss_fn(outputs, targets)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'Batch {i}: training loss = {running_tloss / 10:.4f}')
            running_tloss = 0

    # --- Evaluation ---
    model.eval()
    running_vloss = 0.0
    print("Evaluating on validation data...")

    with torch.no_grad():
        for i, data in enumerate(tqdm(v_loader)):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device).float()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_vloss += loss.item()

            # Visualize the first 3 images in the validation set
            if i < 1:  # visualize only first batch
                for j in range(min(3, inputs.shape[0])):
                    img = inputs[j].cpu()
                    gt = targets[j].cpu().numpy()
                    pred = outputs[j].cpu().numpy()
                    draw_bounding_boxes(img, gt, pred)

        avg_loss = running_vloss / len(v_loader)
        print(f'Average validation loss: {avg_loss:.4f}')

torch.save(model.state_dict(), model_path)
print(f'Model saved to: {model_path}')
