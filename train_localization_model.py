import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.patches as patches
from road_sign_detection.models.localization_base_model import BboxRegression
from road_sign_detection.data.dataset_loader import TT100KDataset
from road_sign_detection.data.annotations import check_annotations
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


class KFoldTrainer:
    def __init__(self, root: str, k_splits: int = 5, epochs: int = 5, batch_size: int = 4, lr: float = 0.001):
        self.root = root
        self.epochs = epochs
        self.k_splits = k_splits
        self.batch_size = batch_size
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_base_path = os.path.join(os.getcwd(), 'models', 'localization_model')
        os.makedirs(os.path.dirname(self.model_base_path), exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

        self.root = root
        self._filter_annotations()
        self.filtered_annotations_path = os.path.join(
            root, 'train_val_annotations.json'
        )

        self.dataset = TT100KDataset(self.filtered_annotations_path, root, self.transform)

    def _filter_annotations(self):
        check_annotations(self.root)

    def train(self):
        kf = KFold(n_splits=self.k_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            print(f"\n=== Fold {fold_idx + 1}/{self.k_splits} ===")

            train_loader = DataLoader(Subset(self.dataset, train_idx),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

            val_loader = DataLoader(Subset(self.dataset, val_idx),
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    collate_fn=collate_fn)

# Initialize training loop parameters
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
        outputs = model(images)

        print(f"Output: {outputs[0].detach().cpu().numpy()}")
        print(f"Label : {labels[0].detach().cpu().numpy()}")
        t_loss = loss_fn(outputs, labels)
        t_loss.backward()
        running_tloss += t_loss.item()
        optimizer.step()

        if i % 10 == 0:
            print(f'\nBatch {i}: Avg Loss = {running_tloss / 10:.6f}')
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

torch.save(model.state_dict(), model_path)
print(f'Saved model to: {model_path}')
model = BboxRegression()
model = model.to(device)
m_state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(m_state_dict)

# Inference on a single sample
model.eval()
sample_image, sample_target = train_split.__getitem__(4)
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
