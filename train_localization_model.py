import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.patches as patches
from project_name.models.localization_base_model import BboxRegression
from project_name.data.dataset_loader import TT100KDataset
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple, List, Dict


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
               ) -> Tuple[Tuple[torch.Tensor, ...],
                          Tuple[Dict[str, torch.Tensor], ...]]:
    """
    Collate function to handle batches of variable-size data.

    Args:
        batch (List): A list of (image, target) pairs.

    Returns:
        Tuple of images and targets as separate tuples.
    """
    return tuple(zip(*batch))


# Set up paths
root: str = os.getcwd() + '/data_storage/tt100k_2021/'
annotations: str = root + 'annotations_all.json'
filtered_annotations: str = root + 'filtered_annotations.json'
ids_file: str = root + 'train/ids.txt'
model_path: str = os.getcwd() + '/models/localization_model.pth'

# Filter annotations by train IDs
if not os.path.exists(filtered_annotations):
    print('Creating a new .json file for training ids.')
    with open(ids_file, 'r') as f:
        ids = set(line.strip() for line in f)

    with open(annotations, 'r') as f:
        annos = json.load(f)

    filtered_imgs = {
        img_id: img_data
        for img_id, img_data in annos['imgs'].items()
        if img_id in ids
    }

    print(f'Found {len(filtered_imgs)} training images.')

    train_annotations = {
        'types': annos['types'],
        'imgs': filtered_imgs
    }

    with open(filtered_annotations, 'w') as f:
        json.dump(train_annotations, f, indent=4)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# Load and split dataset
tt100k_data = TT100KDataset(filtered_annotations, root, transform)
t_size: int = int(0.8 * len(tt100k_data))
v_size: int = len(tt100k_data) - t_size
train_split, val_split = random_split(tt100k_data, [t_size, v_size])

t_loader = DataLoader(train_split,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=collate_fn)
v_loader = DataLoader(val_split,
                      batch_size=4,
                      shuffle=False,
                      collate_fn=collate_fn)

# Training setup
epochs: int = 1
lr: float = 0.001
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')

model = BboxRegression().to(device)
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    print(f'Epoch [{epoch + 1}/{epochs}]')
    model.train()
    running_tloss: float = 0.0

    for i, (images, targets) in enumerate(tqdm(t_loader)):
        images = list(images)
        labels = [t['boxes'] for t in targets]
        images_tensor = torch.stack(images).to(device)
        labels_tensor = torch.stack(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images_tensor)
        t_loss = loss_fn(outputs, labels_tensor)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'\nBatch {i}: Avg Loss = {running_tloss / 10:.4f}')
            running_tloss = 0.0

    # Validation loop
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        for images, targets in tqdm(v_loader):
            images = list(images)
            labels = [t['boxes'] for t in targets]
            images_tensor = torch.stack(images).to(device)
            labels_tensor = torch.stack(labels).to(device)
            outputs = model(images_tensor)
            v_loss = loss_fn(outputs, labels_tensor)
            val_loss += v_loss.item()

    print(f'Validation Loss: {val_loss / len(v_loader):.4f}')

# Save trained model
torch.save(model.state_dict(), model_path)
print(f'Saved model to: {model_path}')

# Inference on a single sample
model.eval()
sample_image, sample_target = val_split[0]
image_tensor = sample_image.to(device).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor).squeeze(0)

# Convert image back to numpy
denorm_image = inv_normalize(image_tensor).squeeze(0)
npimg = denorm_image.cpu().numpy().transpose(1, 2, 0)

# Plot prediction vs ground truth
fig, ax = plt.subplots(1)
ax.imshow(npimg)

x1, y1, x2, y2 = prediction.cpu() * torch.tensor([224, 224, 224, 224])
w, h = x2 - x1, y2 - y1
rect_pred = patches.Rectangle((x1, y1), w, h,
                              linewidth=2,
                              edgecolor='red',
                              facecolor='none')
ax.add_patch(rect_pred)

x1, y1, x2, y2 = sample_target['boxes'] * torch.tensor([224, 224, 224, 224])
w, h = x2 - x1, y2 - y1
rect_gt = patches.Rectangle((x1, y1), w, h,
                            linewidth=2,
                            edgecolor='green',
                            facecolor='none')
ax.add_patch(rect_gt)

plt.title("Red: Prediction | Green: Ground Truth")
plt.show()
