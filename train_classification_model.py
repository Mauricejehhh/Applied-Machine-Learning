"""
Dataset folder should be located under
Applied-Machine-Learning/road_sign_detection/data/tt100k_2021.
Make sure this is the case, otherwise it will not work.
"""
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset
from road_sign_detection.data.annotations import check_annotations


class DatasetPreparer:
    """
    Handles creation of filtered annotation JSON file based on train IDs.
    """

    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        self.annos_path = os.path.join(data_root, 'annotations_all.json')
        self.filtered_path = os.path.join(data_root, 'train_val_annotations.json')

    def prepare(self) -> str:
        check_annotations(self.data_root)
        return self.filtered_path


class DataModule:
    """
    Handles dataset loading, transformation, and splitting.
    """

    def __init__(self, annotation_path: str,
                 data_root: str,
                 batch_size: int = 32) -> None:
        self.annotation_path = annotation_path
        self.data_root = data_root
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def setup(self) -> Tuple[DataLoader, DataLoader, int]:
        """
        Sets up training and validation DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader, int]: train_loader, val_loader, num_classes
        """
        dataset = TT100KSignDataset(self.annotation_path,
                                    self.data_root,
                                    self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        return (
            DataLoader(train_data, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_data, batch_size=self.batch_size, shuffle=True),
            len(dataset.annotations['types'])
        )


class Trainer:
    """
    Encapsulates training and validation routines for a model.
    """

    def __init__(self, model: nn.Module,
                 device: str,
                 learning_rate: float = 0.001) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, d_loader: DataLoader, ep_idx: int) -> None:
        """
        Trains the model for one epoch.

        Args:
            d_loader (DataLoader): Training DataLoader.
            ep_idx (int): Current epoch index.
        """
        self.model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(d_loader,
                                                  desc=f"Epoch {ep_idx+1} Training")):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f'\nBatch {i}: Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    def validate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Validates the model on the provided validation dataset.

        Args:
            data_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            Tuple[float, float]: A tuple containing:
                - Average validation loss (float)
                - Validation accuracy (float)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total
        print(f'Validation Loss: {avg_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')
        return avg_loss, accuracy


class TrainingPipeline:
    """
    Full training pipeline that ties together dataset prep, data loading, training, and saving.
    """

    def __init__(self, data_root:
                 str, model_save_path:
                 str, epochs: int = 1) -> None:
        self.data_root = data_root
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self) -> None:
        """
        Executes the training pipeline:
        - Prepares filtered dataset
        - Loads data
        - Trains and validates model
        - Saves trained weights
        """
        preparer = DatasetPreparer(self.data_root)
        annotation_path = preparer.prepare()

        data_module = DataModule(annotation_path, self.data_root)
        train_loader, val_loader, num_classes = data_module.setup()

        model = CNNClassifier(num_classes)
        trainer = Trainer(model, self.device)

        for epoch in range(self.epochs):
            trainer.train(train_loader, epoch)
            val_loss, val_accuracy = trainer.validate(val_loader)
            random_accuracy = 1 / num_classes
            print(f'Random Guess Accuracy: {random_accuracy:.4f}')
            if val_accuracy > random_accuracy:
                print(" Model performs better than random guessing.")
            else:
                print(" Model is not yet better than random guessing.")

        torch.save(model.state_dict(), self.model_save_path)
        print(f'Model saved to: {self.model_save_path}')


if __name__ == "__main__":
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    model_path = os.path.join(os.getcwd(), 'models', 'model.pth')
    pipeline = TrainingPipeline(root_path, model_path, epochs=1)
    pipeline.run()
