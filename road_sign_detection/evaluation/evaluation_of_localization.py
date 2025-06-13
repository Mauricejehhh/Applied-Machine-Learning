import os
from typing import Tuple, List

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from road_sign_detection.models.localization_base_model import BboxRegression
from road_sign_detection.data.dataset_loader import TT100KDataset
from road_sign_detection.data.annotations import check_annotations


def collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple:
    """
    Custom collate function to group image and target pairs into batches.

    Args:
        batch: A list of (image, target) tuples.

    Returns:
        Tuple of images and targets grouped by batch.
    """
    return tuple(zip(*batch))


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box [x1, y1, x2, y2].
        box2: Second bounding box [x1, y1, x2, y2].

    Returns:
        IoU value between 0 and 1.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def clamp_box(box: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """
    Clamp bounding box coordinates to be within [0, max_val].

    Args:
        box: Bounding box coordinates.
        max_val: Maximum value (default is 1.0 for normalized coordinates).

    Returns:
        Clamped bounding box.
    """
    return np.clip(box, 0, max_val)


def get_device() -> str:
    """
    Get the computation device to be used.

    Returns:
        'cuda' if available, else 'cpu'.
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_dataloader(annotations_path: str, root_dir: str) -> DataLoader:
    """
    Create a DataLoader for the TT100K dataset.

    Args:
        annotations_path: Path to JSON annotation file.
        root_dir: Root directory containing the images.

    Returns:
        DataLoader object.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = TT100KDataset(annotations_path, root_dir, transform)
    return DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


def load_model(model_path: str, device: str) -> torch.nn.Module:
    """
    Load a pretrained bounding box regression model.

    Args:
        model_path: Path to the model file.
        device: Device to map the model to.

    Returns:
        Loaded and ready-to-evaluate model.
    """
    model = BboxRegression().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """
    Evaluate the model and compare it against a random baseline using IoU.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader with test dataset.
        device: Computation device ('cpu' or 'cuda').

    Returns:
        Tuple of (mean IoU for model, mean IoU for random baseline).
    """
    ious_model: List[float] = []
    ious_random: List[float] = []

    for images, targets in tqdm(dataloader, desc="Evaluating"):
        images_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            preds = model(images_tensor).cpu()

        for pred, target in zip(preds, targets):
            gt_box = clamp_box(target['boxes'].numpy())
            pred_box = clamp_box(pred.numpy())

            ious_model.append(compute_iou(pred_box, gt_box))

            rand_x1, rand_y1 = np.random.uniform(0.0, 0.6, 2)
            rand_w, rand_h = np.random.uniform(0.2, 0.4, 2)
            rand_box = clamp_box(np.array([rand_x1, rand_y1, rand_x1 + rand_w, rand_y1 + rand_h]))
            ious_random.append(compute_iou(rand_box, gt_box))

    return float(np.mean(ious_model)), float(np.mean(ious_random))


def main() -> None:
    """
    Main entry point: evaluates bounding box regression model on TT100K dataset.
    Prints IoU results for both the model and a random baseline.
    """
    root = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    check_annotations(root)

    annotations_path = os.path.join(root, 'test_annotations.json')
    model_path = os.path.join(os.getcwd(), 'models', 'localization_model_final_ensemble.pth')

    device = get_device()
    print(f"Using device: {device}")

    dataloader = get_dataloader(annotations_path, root)
    model = load_model(model_path, device)
    mean_iou_model, mean_iou_random = evaluate(model, dataloader, device)

    print(f"\nMean IoU (Model) : {mean_iou_model:.4f}")
    print(f"Mean IoU (Random): {mean_iou_random:.4f}")

    if mean_iou_model > mean_iou_random:
        print("=> Model outperforms random guessing.")
    else:
        print("=> Model does not outperform random guessing.")


if __name__ == "__main__":
    main()
