import os
from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter

from road_sign_detection.models.classification_base_model import CNNClassifier
from road_sign_detection.data.dataset_loader import TT100KSignDataset
from road_sign_detection.data.annotations import check_annotations


class DatasetPreparer:
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        self.annos_path = os.path.join(data_root, 'annotations_all.json')
        self.filtered_path = os.path.join(data_root, 'train_val_annotations.json')

    def prepare(self) -> str:
        check_annotations(self.data_root)
        return self.filtered_path


class DataModule:
    def __init__(self,
                 annotation_path: str,
                 data_root: str,
                 batch_size: int = 32,
                 num_folds: int = 5):
        self.annotation_path = annotation_path
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_folds = num_folds

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

    def get_dataset(self):
        return TT100KSignDataset(self.annotation_path, self.data_root, self.transform)

    def get_folds(self, dataset):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        return list(kf.split(dataset))


class Trainer:
    def __init__(self, model: nn.Module, device: str, learning_rate: float = 0.001) -> None:
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, d_loader: DataLoader, ep_idx: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            d_loader (DataLoader): Training DataLoader.
            ep_idx (int): Current epoch index.
        """
        self.model.train()
        running_loss = 0.0
        running_total_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(
            d_loader,
            desc=f"Epoch {ep_idx + 1} Training")
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_total_loss += loss.item()

            if i % 10 == 0:
                print(f'\nBatch {i}: Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Calculate average loss over entire epoch
        avg_loss = running_total_loss / len(d_loader)
        print(f'Epoch {ep_idx + 1} Average Training Loss: {avg_loss:.4f}')

        return avg_loss

    def validate(self, data_loader: DataLoader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total
        print(f'Validation Loss: {avg_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')

        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


class TrainingPipeline:
    """ Full training pipeline that ties together dataset prep,
    data loading, training, and saving.
    """

    def __init__(self, data_root:
                 str, model_save_path:
                 str, epochs: int = 1) -> None:
        self.data_root = data_root
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def average_state_dicts(state_dicts):
        avg_state_dict = {}
        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
        return avg_state_dict

    def plot_losses(self, train_losses_all_folds, val_losses_all_folds):
        for fold_idx, (train_losses, val_losses) in enumerate(
            zip(train_losses_all_folds, val_losses_all_folds)
        ):
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o')
            plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
            plt.title(f'Fold {fold_idx + 1} Train/Val Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    # def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: list, top_n: int = 50):
    #         class_counts = Counter(y_true)
    #         most_common = [cls for cls, _ in class_counts.most_common(top_n)]

    #         mask = np.isin(y_true, most_common)
    #         y_true_filtered = y_true[mask]
    #         y_pred_filtered = y_pred[mask]

    #         labels = sorted(most_common)
    #         cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
    #         cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize

    #         subset_class_names = [class_names[i] for i in labels]
    #         plt.figure(figsize=(18, 14))
    #         sns.heatmap(cm, annot=False, fmt=".2f", cmap="jet",
    #                     xticklabels=subset_class_names, yticklabels=subset_class_names)

    #         plt.title(f"Top-{top_n} Most Common Classes - Confusion Matrix")
    #         plt.xlabel("Predicted Label")
    #         plt.ylabel("True Label")
    #         plt.xticks(rotation=90)
    #         plt.yticks(rotation=0)
    #         plt.tight_layout()
    #         plt.show()


    def run(self) -> None:
        preparer = DatasetPreparer(self.data_root)
        annotation_path = preparer.prepare()

        import json
        with open(annotation_path, 'r') as f:
            annos = json.load(f)

        # Check if 'types' is dict or list
        if isinstance(annos['types'], dict):
            class_names = list(annos['types'].keys())
        elif isinstance(annos['types'], list):
            class_names = annos['types']
        else:
            raise ValueError(f"Unexpected type for 'types': {type(annos['types'])}")

        num_classes = len(class_names)

        data_module = DataModule(annotation_path, self.data_root, num_folds=2)
        dataset = data_module.get_dataset()
        folds = data_module.get_folds(dataset)

        all_accuracies = []
        all_state_dicts = []

        train_losses_all_folds = []
        val_losses_all_folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{len(folds)} ---")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset,
                                    batch_size=data_module.batch_size,
                                    shuffle=True)
            val_loader = DataLoader(val_subset,
                                    batch_size=data_module.batch_size,
                                    shuffle=False)

            model = CNNClassifier(num_classes)
            trainer = Trainer(model, self.device)

            train_losses = []
            val_losses = []

            for epoch in range(self.epochs):
                train_loss = trainer.train(train_loader, epoch)
                val_loss, val_accuracy, val_preds, val_labels = trainer.validate(val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

            train_losses_all_folds.append(train_losses)
            val_losses_all_folds.append(val_losses)

            all_accuracies.append(val_accuracy)
            all_state_dicts.append(model.state_dict())

            # self.plot_confusion_matrix(val_labels, val_preds, class_names)

            fold_model_path = self.model_save_path.replace('.pth', f'_fold{fold_idx + 1}.pth')
            torch.save(model.state_dict(), fold_model_path)
            print(f'Model for fold {fold_idx+1} saved to: {fold_model_path}')

        avg_accuracy = sum(all_accuracies) / len(all_accuracies)
        print(f"\nAverage Cross-Validation Accuracy: {avg_accuracy:.4f}")

        # Average weights from all folds
        print("\nAveraging model weights across folds...")
        averaged_state_dict = self.average_state_dicts(all_state_dicts)

        final_model = CNNClassifier(num_classes)
        final_model.load_state_dict(averaged_state_dict)
        torch.save(final_model.state_dict(), self.model_save_path)
        print(f"Final averaged model saved to: {self.model_save_path}")

        # Plot the losses per fold
        self.plot_losses(train_losses_all_folds, val_losses_all_folds)


if __name__ == "__main__":
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    model_path = os.path.join(os.getcwd(), 'models', 'classification_model_2.pth')
    pipeline = TrainingPipeline(root_path, model_path, epochs=1)
    pipeline.run()
