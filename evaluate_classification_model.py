"""
Evaluation script for a trained CNN classifier on the TT100K dataset.

Steps:
- Prepare dataset annotations (filter test set).
- Load and preprocess test dataset.
- Load trained model.
- Run evaluation and display sample predictions.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Set, Dict, Any, Tuple

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset


def create_filtered_annotations(
    annotations_path: str,
    ids_path: str,
    output_path: str
) -> None:
    """
    Create a filtered annotation JSON file based on a set of image IDs.

    Args:
        annotations_path (str): Path to original annotations JSON.
        ids_path (str): Path to the file containing image IDs to keep.
        output_path (str): Path where the filtered annotations will be saved.
    """
    print('Creating a new .json file for test ids.')
    with open(ids_path, 'r') as f:
        ids: Set[str] = set(line.strip() for line in f)

    with open(annotations_path, 'r') as f:
        annotations: Dict[str, Any] = json.load(f)

    filtered_imgs = {
        img_id: img_data for img_id, img_data in annotations['imgs'].items() if img_id in ids
    }

    filtered_annotations = {
        'types': annotations['types'],
        'imgs': filtered_imgs
    }

    with open(output_path, 'w') as f:
        json.dump(filtered_annotations, f, indent=4)


def get_test_loader(annotations_file: str,
                    root: str) -> Tuple[DataLoader, TT100KSignDataset]:
    """
    Load the test dataset and return a DataLoader and the dataset object.

    Args:
        annotations_file (str): Path to filtered annotations file.
        root (str): Root directory for images.

    Returns:
        Tuple[DataLoader, TT100KSignDataset]: The test DataLoader and dataset object.
    """
    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = TT100KSignDataset(annotations_file, root, transform_pipeline)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader, dataset


def load_model(model_path: str,
               num_classes: int,
               device: torch.device) -> CNNClassifier:
    """
    Load a trained model from a given path.

    Args:
        model_path (str): Path to the saved model weights.
        num_classes (int): Number of output classes.
        device (torch.device): Torch device to load model onto.

    Returns:
        CNNClassifier: The loaded model.
    """
    model = CNNClassifier(num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def matplotlib_imshow(img: torch.Tensor, one_channel: bool = False) -> None:
    """
    Display a tensor image using matplotlib.

    Args:
        img (torch.Tensor): Image tensor to display.
        one_channel (bool): Whether to convert image to grayscale for display.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def evaluate(model: CNNClassifier,
             dataloader: DataLoader,
             device: torch.device) -> float:
    """
    Evaluate the model on the test set.

    Args:
        model (CNNClassifier): Trained model.
        dataloader (DataLoader): Test data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Accuracy as a percentage.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def show_sample_predictions(model: CNNClassifier,
                            dataset: TT100KSignDataset,
                            device: torch.device) -> None:
    """
    Display predictions for one batch of test data.

    Args:
        model (CNNClassifier): Trained model.
        dataset (TT100KSignDataset): Dataset used for label mapping.
        device (torch.device): Device for inference.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)

    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    plt.show()

    print("Ground Truth:",
          [dataset.idx_to_label[label.item()] for label in labels])
    print("Predicted   :",
          [dataset.idx_to_label[p.item()] for p in predicted])


def main() -> None:
    """Main evaluation entry point."""
    root = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    annotations_path = os.path.join(root, 'annotations_all.json')
    filtered_annotations = os.path.join(root, 'filtered_test_annotations.json')
    ids_file = os.path.join(root, 'test', 'ids.txt')
    model_path = os.path.join(os.getcwd(), 'models', 'classi_model.pth')

    if not os.path.exists(filtered_annotations):
        create_filtered_annotations(annotations_path, ids_file,
                                    filtered_annotations)

    test_loader, test_dataset = get_test_loader(filtered_annotations, root)

    print(f'Cuda (GPU support) available: {torch.cuda.is_available()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, len(test_dataset.labels), device)
    model.eval()

    show_sample_predictions(model, test_dataset, device)
    accuracy = evaluate(model, test_loader, device)
    print(f'Accuracy on testing data: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
