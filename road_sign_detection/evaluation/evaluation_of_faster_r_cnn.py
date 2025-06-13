import os
from typing import List, Tuple, Dict

import torch
from torch import Tensor
from torch.utils.data import Subset, DataLoader
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm

from road_sign_detection.data.dataset_loader import TT100KFRCNNDataset
from road_sign_detection.data.annotations import check_annotations


def get_transform() -> T.Compose:
    """
    Returns the transformation to apply to each image in the dataset.
    """
    return T.Compose([T.ToTensor()])


def collate_fn(batch):
    """
    Custom collate function for object detection batches.
    """
    return tuple(zip(*batch))


def get_model(num_classes: int, model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads a Faster R-CNN model with the given number of classes and loads pre-trained weights.

    Args:
        num_classes (int): Number of object classes including background.
        model_path (str): Path to the model checkpoint.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compute_metrics(
    preds: List[Dict[str, Tensor]],
    targets: List[Dict[str, Tensor]]
) -> Tuple[float, float]:
    """
    Computes classification accuracy and mean Area of Intersection (AoI) between predicted and ground truth boxes.

    Args:
        preds (List[Dict[str, Tensor]]): List of model predictions with 'boxes' and 'labels'.
        targets (List[Dict[str, Tensor]]): Corresponding ground truth annotations.

    Returns:
        Tuple[float, float]: Classification accuracy and mean AoI.
    """
    correct_classifications = 0
    total_classifications = 0
    total_aoi = 0
    matched_boxes = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        ious = box_iou(gt_boxes, pred_boxes)
        max_ious, max_indices = ious.max(dim=1)

        for i, iou in enumerate(max_ious):
            if iou > 0.5:
                matched_boxes += 1
                total_aoi += iou.item()

                if gt_labels[i] == pred_labels[max_indices[i]]:
                    correct_classifications += 1
                total_classifications += 1

    accuracy = correct_classifications / total_classifications if total_classifications else 0
    mean_aoi = total_aoi / matched_boxes if matched_boxes else 0
    return accuracy, mean_aoi


def main():
    """
    Main function to evaluate a Faster R-CNN model on the TT100K dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset setup
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    check_annotations(root_path)
    annotations_file = os.path.join(root_path, 'train_val_annotations.json')
    dataset = TT100KFRCNNDataset(annotations_file=annotations_file, root_dir=root_path, transform=get_transform())
    num_classes = len(dataset.label_to_idx) + 1  # +1 for background class

    # Model setup
    model_path = os.path.join('models', 'frcnn', 'frcnn_final_averaged.pth')
    model = get_model(num_classes, model_path, device)

    # Subset for evaluation
    max_images = 300
    indices = list(range(min(len(dataset), max_images)))
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    all_preds: List[Dict[str, Tensor]] = []
    all_targets: List[Dict[str, Tensor]] = []

    # Evaluation loop
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            outputs = model(images)
            all_preds.extend([{k: v.cpu() for k, v in out.items()} for out in outputs])
            all_targets.extend([{k: v.cpu() for k, v in tgt.items()} for tgt in targets])

    # Metrics
    acc, mean_aoi = compute_metrics(all_preds, all_targets)

    print(f"\n--- Evaluation Results ---")
    print(f"Classification Accuracy: {acc * 100:.2f}%")
    print(f"Mean Area of Intersection (AoI): {mean_aoi:.4f}")


if __name__ == "__main__":
    main()
