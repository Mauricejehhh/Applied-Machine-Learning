import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset
from road_sign_detection.data.annotations import check_annotations


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: torch.device
) -> None:
    model.train()  # Set model in training mode
    running_loss = 0.0

    for i, (images, labels_truth) in enumerate(tqdm(dataloader)):
        images, labels_truth = images.to(device), labels_truth.to(device)
        optimizer.zero_grad()
        label_pred = model(images)
        loss = loss_fn(label_pred, labels_truth)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Training loss: {(running_loss / len(dataloader)):.4f}')


def validate_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss.CrossEntropyLoss,
    device: torch.device
) -> None:
    model.eval()  # Set model in evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, labels_truth) in enumerate(tqdm(dataloader)):
            images, labels_truth = images.to(device), labels_truth.to(device)
            label_pred = model(images)
            loss = loss_fn(label_pred, labels_truth)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(dataloader)
    print(f'Average validation loss: {avg_val_loss:.4f}')


if __name__ == '__main__':
    # Define paths and find/create annotations
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')

    # Splits default to 0.70/0.15/0.15, set the splits as arguments
    check_annotations(root_path)
    train_path = os.path.join(root_path, 'train_val_annotations.json')

    # Define transform and initialize dataset
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

    data = TT100KSignDataset(train_path, root_path, transform)

    train_size = int(0.70 / 0.85 * len(data))
    val_size = len(data) - train_size
    assert train_size + val_size == len(data), \
        'Sum of the train and val splits is not the same as the original size.'

    train_split, val_split = random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_split, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=16, shuffle=False)

    lr = 0.0001

    model = CNNClassifier(len(data.labels))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    epochs = 20
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        validate_one_epoch(val_loader, model, loss_fn, device)
