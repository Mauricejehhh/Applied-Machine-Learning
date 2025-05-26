import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from project_name.models.classification_base_model import CNNClassifier
from project_name.models.localization_base_model import cnn_model
from project_name.data.dataset_loader import TT100KDataset
from torch.utils.data import random_split, DataLoader


# These work for me, I am unsure whether they could work for you.
# To be sure, run this file from the /Applied-Machine-Learning
# directory, it should theoretically work.
root = os.getcwd() + '/data_storage/tt100k_2021/'
annotations = root + 'annotations_all.json'
filtered_annotations = root + 'filtered_annotations.json'
ids_file = root + 'train/ids.txt'
model_path = os.getcwd() + '/models/model.pth'


# Find all training ids from the ids.txt file in train/
# This is bound to change as custom splits will be needed.
if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    # Filter annotations file for training ids only
    filtered_imgs = {img_id: img_data
                     for img_id, img_data in annos['imgs'].items()
                     if img_id in ids}
    print(f'Found {len(filtered_imgs)} training images.')

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

# Set up transforms
transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
tt100k_data = TT100KDataset(filtered_annotations, root, transforms)
t_size = int(0.8 * len(tt100k_data))
v_size = len(tt100k_data) - t_size
train_split, val_split = random_split(tt100k_data, [t_size, v_size])

t_loader = DataLoader(train_split, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
v_loader = DataLoader(val_split, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Device and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cnn_model().to(device)

# Use SmoothL1Loss for bounding box regression
loss_fn = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training loop
for epoch in range(1):
    print(f'Epoch [{epoch + 1}/1]')
    model.train()
    running_tloss = 0.0

    for i, (images, targets) in enumerate(tqdm(t_loader)):
        images = [img.to(device) for img in images]
        labels = [t['boxes'].to(device) for t in targets]
        images = torch.stack(images)  # [B, 1, 64, 64]
        labels = torch.stack(labels)  # [B, 4]

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_tloss += loss.item()

        if i % 10 == 0:
            print(f'Batch {i}: Avg Loss = {running_tloss / 10:.4f}')
            running_tloss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in v_loader:
            images = [img.to(device) for img in images]
            labels = [t['boxes'].to(device) for t in targets]
            images = torch.stack(images)
            labels = torch.stack(labels)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    print(f'Validation loss: {val_loss / len(v_loader):.4f}')

torch.save(model.state_dict(), model_path)
print(f'Saved model to: {model_path}')

# Pick one sample from the validation set
sample_image, sample_target = val_split[0]
model.eval()

# Get original image dimensions
_, orig_height, orig_width = sample_image.shape

# Prepare the image
image_tensor = sample_image.unsqueeze(0).to(device)  # Add batch dimension
if image_tensor.shape[1] != 1:
    image_tensor = image_tensor.mean(dim=1, keepdim=True)

# Predict
with torch.no_grad():
    pred = model(image_tensor).cpu().squeeze()  # [4]

# Denormalize predicted bounding box
pred_box = pred * torch.tensor([orig_width, orig_height, orig_width, orig_height])

# Extract and denormalize ground truth bounding box
gt_box_tensor = sample_target['boxes'].squeeze()  # Shape: [4]
gt_box = gt_box_tensor * torch.tensor([orig_width, orig_height, orig_width, orig_height])

# Convert image for plotting
image_np = sample_image.permute(1, 2, 0).numpy()
image_np = image_np * 0.5 + 0.5  # Unnormalize if necessary
image_np = image_np.squeeze() if image_np.shape[2] == 1 else image_np

# Plot
fig, ax = plt.subplots(1)
ax.imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)

# Predicted box (Red)
x1, y1, x2, y2 = pred_box
w, h = x2 - x1, y2 - y1
pred_rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none', label="Prediction")
ax.add_patch(pred_rect)

# Ground truth box (Green)
gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
gt_w, gt_h = gt_x2 - gt_x1, gt_y2 - gt_y1
gt_rect = patches.Rectangle((gt_x1, gt_y1), gt_w, gt_h, linewidth=2, edgecolor='g', facecolor='none', label="Ground Truth")
ax.add_patch(gt_rect)

ax.legend()
plt.title("Red: Predicted | Green: Ground Truth")
plt.show()
