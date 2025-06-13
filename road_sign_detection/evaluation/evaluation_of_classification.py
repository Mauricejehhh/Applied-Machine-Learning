import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset
from road_sign_detection.data.annotations import check_annotations


def get_data_loader(data_root: str, annotation_file: str, batch_size: int = 32) -> Tuple[DataLoader, int]:
    """
    Load and return the test DataLoader and number of classes.
    Args:
        data_root (str): Path to the dataset root.
        annotation_file (str): Path to the annotation file.
        batch_size (int): Batch size for DataLoader.
    Returns:
        Tuple[DataLoader, int]: DataLoader for the test set and number of unique classes.
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    dataset = TT100KSignDataset(annotation_file, data_root, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_classes = len(dataset.annotations['types'])
    return loader, num_classes


def load_model(model_path: str, num_classes: int, device: torch.device) -> CNNClassifier:
    """
    Load the trained model from a checkpoint.
    Args:
        model_path (str): Path to the saved model file.
        num_classes (int): Number of output classes.
        device (torch.device): Torch device (CPU or CUDA).
    Returns:
        CNNClassifier: Loaded CNN model.
    """
    model = CNNClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate(model: CNNClassifier, dataloader: DataLoader, device: torch.device
             ) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate the model on the given dataset.
    Args:
        model (CNNClassifier): Trained CNN model.
        dataloader (DataLoader): DataLoader with the test dataset.
        device (torch.device): Torch device (CPU or CUDA).
    Returns:
        Tuple[float, float, List[int], List[int]]: Average loss, accuracy, all predictions, all true labels.
    """
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def main():
    """
    Main entry point for model evaluation.
    """
    root = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    check_annotations(root)

    annotations = os.path.join(root, 'test_annotations.json')
    model_path = os.path.join(os.getcwd(), 'models', 'classification_model_2_fold2.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_loader, num_classes = get_data_loader(root, annotations)
    model = load_model(model_path, num_classes, device)
    avg_loss, accuracy, all_preds, all_labels = evaluate(model, test_loader, device)

    print(f"\nTest Loss       : {avg_loss:.4f}")
    print(f"Test Accuracy   : {accuracy:.4f}")

    random_guess_acc = 1 / num_classes
    print(f"Random Guess Accuracy: {random_guess_acc:.4f}")

    if accuracy > random_guess_acc:
        print("=> Model outperforms random guessing.")
    else:
        print("=> Model does not outperform random guessing.")


if __name__ == "__main__":
    main()
