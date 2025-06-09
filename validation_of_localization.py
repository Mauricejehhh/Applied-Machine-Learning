import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, List
from tqdm import tqdm

from road_sign_detection.models.localization_base_model import BboxRegression
from road_sign_detection.data.dataset_loader import TT100KDataset
from road_sign_detection.data.annotations import check_annotations


def collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple:
    """
    Custom collate function to unpack dataset samples into batch tuples.

    Args:
        batch: A list of (image, target) tuples.

    Returns:
        Tuple of images and targets.
    """
    return tuple(zip(*batch))


# ------------------- Configuration -------------------

# Paths
root = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
check_annotations(root)
annotations = os.path.join(root, 'test_annotations.json')
model_path = os.path.join(os.getcwd(), 'models', 'localization_model_final_ensemble.pth')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------- Dataset Preparation -------------------

# Load dataset and split
dataset = TT100KDataset(annotations, root, transform)
val_size: int = int(0.2 * len(dataset))
train_size: int = len(dataset) - val_size
_, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

val_loader = DataLoader(
    val_set,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------- Model Setup -------------------

# Choose device
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load trained model
model = BboxRegression().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# ------------------- Evaluation Utilities -------------------

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU as a float between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# ------------------- Evaluation Loop -------------------

ious_model: List[float] = []
ious_random: List[float] = []

for images, targets in tqdm(val_loader, desc="Evaluating"):
    images_tensor = torch.stack(images).to(device)

    with torch.no_grad():
        preds = model(images_tensor).cpu()

    for pred, target in zip(preds, targets):
        gt: torch.Tensor = target['boxes']
        pred_np: np.ndarray = pred.numpy()
        gt_np: np.ndarray = gt.numpy()

        # Model-predicted IoU
        iou_pred = compute_iou(pred_np, gt_np)
        ious_model.append(iou_pred)

        # Random box IoU
        rand_x1, rand_y1 = np.random.rand(2) * 0.8
        rand_w, rand_h = np.random.rand(2) * 0.2
        rand_box = np.array([rand_x1, rand_y1, rand_x1 + rand_w, rand_y1 + rand_h])
        iou_rand = compute_iou(rand_box, gt_np)
        ious_random.append(iou_rand)

# ------------------- Results -------------------

mean_iou_model = float(np.mean(ious_model))
mean_iou_random = float(np.mean(ious_random))

print(f"\nMean IoU (Model) : {mean_iou_model:.4f}")
print(f"Mean IoU (Random): {mean_iou_random:.4f}")
if mean_iou_model > mean_iou_random:
    print("=> Model outperforms random guessing.")
else:
    print("=> Model does not outperform random guessing.")
