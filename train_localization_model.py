import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import List, Tuple, Dict

from road_sign_detection.models.localization_base_model import BboxRegression
from road_sign_detection.data.dataset_loader import TT100KDataset
from road_sign_detection.data.annotations import check_annotations


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
               ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
    return tuple(zip(*batch))


class KFoldTrainer:
    def __init__(self, data_root: str, k_splits: int = 5, epochs: int = 5, batch_size: int = 4, lr: float = 0.001):
        self.data_root = data_root
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

        self._filter_annotations()
        self.train_annotations = os.path.join(data_root, 'train_val_annotations.json')
        self.dataset = TT100KDataset(self.train_annotations, data_root, self.transform)

    def _filter_annotations(self):
        check_annotations(self.data_root)

    def train(self):
        kf = KFold(n_splits=self.k_splits, shuffle=True, random_state=42)

        train_losses_all_folds = []
        val_losses_all_folds = []

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

            model = BboxRegression().to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.SmoothL1Loss()

            train_losses = []
            val_losses = []

            for epoch in range(self.epochs):
                print(f"\nEpoch [{epoch + 1}/{self.epochs}]")
                avg_train_loss = self._train_one_epoch(model, train_loader, loss_fn, optimizer)
                avg_val_loss = self._validate(model, val_loader, loss_fn)
                print(f'Fold {fold_idx + 1} Train Loss: {avg_train_loss:.4f}')
                print(f'Fold {fold_idx + 1} Validation Loss: {avg_val_loss:.4f}')

                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
            
            train_losses_all_folds.append(train_losses)
            val_losses_all_folds.append(val_losses)

            model_path = f"{self.model_base_path}_fold{fold_idx + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        self._plot_losses(train_losses_all_folds, val_losses_all_folds)

    def _plot_losses(self, train_losses_all_folds, val_losses_all_folds):
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

    def _train_one_epoch(self, model, dataloader, loss_fn, optimizer):
        model.train()
        running_loss = 0.0
        running_total_loss = 0.0

        for i, (images, targets) in enumerate(tqdm(dataloader)):
            images = torch.stack(images).to(self.device)
            labels = torch.stack([t['boxes'] for t in targets]).to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_total_loss += loss.item()

            if i % 10 == 0 and i > 0:
                print(f"Batch {i}: Avg Loss = {running_loss / 10:.6f}")
                running_loss = 0.0

        return running_total_loss / len(dataloader)

    def _validate(self, model, dataloader, loss_fn):
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(dataloader):
                images = torch.stack(images).to(self.device)
                labels = torch.stack([t['boxes'] for t in targets]).to(self.device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        return val_loss / len(dataloader)

    def ensemble_predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        predictions = []
        for fold in range(1, self.k_splits + 1):
            model = BboxRegression().to(self.device)
            model.load_state_dict(torch.load(f"{self.model_base_path}_fold{fold}.pth", weights_only=True))
            model.eval()
            with torch.no_grad():
                pred = model(image_tensor).squeeze(0)
                predictions.append(pred)
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred

    def save_ensemble_model(self):
        state_dicts = []
        for fold in range(1, self.k_splits + 1):
            model_path = f"{self.model_base_path}_fold{fold}.pth"
            state_dicts.append(torch.load(model_path))

        avg_state_dict = {}
        for key in state_dicts[0].keys():
            avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / self.k_splits

        final_model_path = f"{self.model_base_path}_final_ensemble.pth"
        torch.save(avg_state_dict, final_model_path)
        print(f"Saved final ensemble model to {final_model_path}")

    def visualize_sample_prediction(self, fold: int = 1, sample_idx: int = 4, use_ensemble: bool = False):
        image, target = self.dataset[sample_idx]
        image_tensor = image.unsqueeze(0).to(self.device)

        if use_ensemble:
            pred = self.ensemble_predict(image_tensor)
        else:
            model = BboxRegression().to(self.device)
            model.load_state_dict(torch.load(f"{self.model_base_path}_fold{fold}.pth", weights_only=True))
            model.eval()
            with torch.no_grad():
                pred = model(image_tensor).squeeze(0)

        denorm = self.inv_normalize(image_tensor).squeeze(0).cpu().numpy().transpose(1, 2, 0)

        fig, ax = plt.subplots(1)
        ax.imshow(denorm)

        pred = pred.cpu() * torch.tensor([224, 224, 224, 224])
        gt = target['boxes'] * torch.tensor([224, 224, 224, 224])

        rect_pred = patches.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                                      linewidth=2, edgecolor='red', facecolor='none')
        rect_gt = patches.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                                    linewidth=2, edgecolor='green', facecolor='none')

        ax.add_patch(rect_pred)
        ax.add_patch(rect_gt)
        plt.title("Red: Ensemble Prediction | Green: Ground Truth" if use_ensemble else "Red: Fold Prediction | Green: Ground Truth")
        plt.show()


if __name__ == "__main__":
    trainer = KFoldTrainer(data_root=os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021'))
    trainer.train()
    trainer.save_ensemble_model()
    trainer.visualize_sample_prediction(use_ensemble=True)
