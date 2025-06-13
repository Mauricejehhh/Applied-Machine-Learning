import os
from typing import List, Tuple, Dict
import torch
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from road_sign_detection.data.dataset_loader import TT100KFRCNNDataset
from road_sign_detection.data.annotations import check_annotations

matplotlib.use('Agg')  # Use non-interactive backend


def get_transform(train: bool) -> T.Compose:
    """
    Returns transformation pipeline for training or inference.

    Args:
        train (bool): Whether to include data augmentation for training.

    Returns:
        torchvision.transforms.Compose: Composed transformation pipeline.
    """
    base_transforms = [T.ToTensor()]
    if train:
        base_transforms.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
        ])
    return T.Compose(base_transforms)


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple:
    """
    Custom collate function to group images and targets.

    Args:
        batch (List): List of (image, target) tuples.

    Returns:
        Tuple: Tuple of batched images and targets.
    """
    return tuple(zip(*batch))


class FasterRCNNKFoldTrainer:
    """
    Trainer for Faster R-CNN with K-Fold cross-validation.
    """

    def __init__(self,
                 dataset: TT100KFRCNNDataset,
                 num_classes: int,
                 model_dir: str = 'models/frcnn',
                 k_folds: int = 5,
                 batch_size: int = 4,
                 epochs: int = 5,
                 lr: float = 1e-3) -> None:
        """
        Initializes trainer with dataset and training configuration.

        Args:
            dataset (TT100KFRCNNDataset): Dataset for object detection.
            num_classes (int): Number of target classes including background.
            model_dir (str): Directory to save models and plots.
            k_folds (int): Number of folds for cross-validation.
            batch_size (int): Batch size for DataLoaders.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
        """
        self.dataset = dataset
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.num_classes = num_classes

    def get_model(self) -> torch.nn.Module:
        """
        Loads a Faster R-CNN model with a custom head.

        Returns:
            torch.nn.Module: Customized Faster R-CNN model.
        """
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def evaluate(self,
                 model: torch.nn.Module,
                 data_loader: DataLoader) -> float:
        """
        Evaluates the model on a validation set.

        Args:
            model (torch.nn.Module): Model to evaluate.
            data_loader (DataLoader): Validation data loader.

        Returns:
            float: Average validation loss.
        """
        model_mode = model.training
        model.train()  # must be in train mode to compute losses
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()

        model.train(model_mode)  # restore original training/eval state
        return total_loss / max(len(data_loader), 1)

    def plot_train_val_losses(self,
                              loss_history: Dict[int, Dict[str, List[float]]],
                              fold: int) -> None:
        """
        Plots and saves the training and validation losses for a fold.

        Args:
            loss_history (dict): Dictionary of training/validation loss histories.
            fold (int): Fold number.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.epochs + 1), loss_history[fold]['train'], label='Train Loss')
        plt.plot(range(1, self.epochs + 1), loss_history[fold]['val'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, f'fold_{fold+1}_loss.png'))
        plt.close()

    def train(self) -> None:
        """
        Runs K-Fold training and saves individual and averaged models.
        """
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        loss_history: Dict[int, Dict[str, List[float]]] = {}
        model_paths: List[str] = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            print(f"\n--- Fold {fold + 1}/{self.k_folds} ---")
            print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")
            model = self.get_model().to(self.device)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.lr)

            train_loader = DataLoader(Subset(self.dataset, train_ids), batch_size=self.batch_size,
                                      shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
            val_loader = DataLoader(Subset(self.dataset, val_ids), batch_size=1,
                                    shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

            loss_history[fold] = {'train': [], 'val': []}

            for epoch in range(self.epochs):
                print(f"Starting epoch {epoch + 1}/{self.epochs}...")
                model.train()
                train_loss = 0.0

                for images, targets in tqdm(train_loader, desc=f"Fold {fold + 1} - Epoch {epoch + 1}"):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    try:
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()
                        train_loss += losses.item()
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print("OOM error: Skipping batch")
                            torch.cuda.empty_cache()
                        else:
                            raise e

                avg_train_loss = train_loss / len(train_loader)
                val_loss = self.evaluate(model, val_loader)

                loss_history[fold]['train'].append(avg_train_loss)
                loss_history[fold]['val'].append(val_loss)

                print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            model_path = os.path.join(self.model_dir, f"frcnn_fold{fold + 1}.pth")
            torch.save(model.state_dict(), model_path)
            model_paths.append(model_path)
            print(f"Saved model for fold {fold + 1}")
            self.plot_train_val_losses(loss_history, fold)

        print("\n--- Averaging Fold Models ---")
        averaged_model = self.get_model().to(self.device)
        averaged_state_dict = averaged_model.state_dict()
        fold_state_dicts = [torch.load(p, map_location=self.device) for p in model_paths]

        for key in averaged_state_dict.keys():
            averaged_state_dict[key] = torch.stack([sd[key] for sd in fold_state_dicts], dim=0).mean(dim=0)

        averaged_model.load_state_dict(averaged_state_dict)
        final_model_path = os.path.join(self.model_dir, "frcnn_final_averaged.pth")
        torch.save(averaged_model.state_dict(), final_model_path)
        print(f"Final averaged model saved to {final_model_path}")


if __name__ == "__main__":
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    check_annotations(root_path)
    train_path = os.path.join(root_path, 'train_val_annotations.json')
    dataset = TT100KFRCNNDataset(annotations_file=train_path, root_dir=root_path, transform=get_transform(train=True))
    num_classes = len(dataset.label_to_idx) + 1
    trainer = FasterRCNNKFoldTrainer(dataset, num_classes=num_classes, k_folds=2, epochs=1)
    trainer.train()
