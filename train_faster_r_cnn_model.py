# frcnn_kfold_trainer.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from road_sign_detection.data.dataset_loader import TT100KFRCNNDataset
from road_sign_detection.data.annotations import check_annotations


def get_transform(train: bool):
    base_transforms = [T.ToTensor()]
    if train:
        base_transforms.extend([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(10),
        ])
    return T.Compose(base_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


class FasterRCNNKFoldTrainer:
    def __init__(self, dataset, num_classes, model_dir='models/frcnn', k_folds=5, batch_size=4, epochs=5, lr=1e-3):
        self.dataset = dataset
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.num_classes = num_classes  # include background class

    def get_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model

    def evaluate(self, model, data_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        return total_loss / len(data_loader)

    def visualize_predictions(self, model, val_dataset, label_map, num_images=3):
        model.eval()
        indices = random.sample(range(len(val_dataset)), num_images)
        for idx in indices:
            img, target = val_dataset[idx]
            with torch.no_grad():
                prediction = model([img.to(self.device)])[0]

            img_np = F.to_pil_image(img)
            plt.figure(figsize=(8, 6))
            plt.imshow(img_np)
            ax = plt.gca()

            # Draw predicted boxes
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score > 0.5:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor='red', linewidth=2))
                    label_name = label_map.get(label.item(), str(label.item()))
                    ax.text(x1, y1, f'{label_name}: {score:.2f}', color='red', fontsize=8)
            plt.axis('off')
            plt.title("Predictions")
            plt.show()

    def plot_train_val_losses(self, loss_history, fold):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.epochs + 1), loss_history[fold]['train'], label='Train Loss')
        plt.plot(range(1, self.epochs + 1), loss_history[fold]['val'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold+1} Loss')
        plt.legend()
        plt.show()

    def train(self):
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        label_map = {v: k for k, v in self.dataset.label_to_idx.items()}
        label_map[0] = "background"
        loss_history = {}

        for fold, (train_ids, val_ids) in enumerate(kfold.split(self.dataset)):
            print(f"\n--- Fold {fold+1}/{self.k_folds} ---")
            model = self.get_model().to(self.device)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.lr)

            train_subset = Subset(self.dataset, train_ids)
            val_subset = Subset(self.dataset, val_ids)

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

            loss_history[fold] = {'train': [], 'val': []}

            for epoch in range(self.epochs):
                model.train()
                train_loss = 0
                for images, targets in tqdm(
                    train_loader,
                    desc=f"Fold {fold+1} - Epoch {epoch+1}"
                ):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()}
                               for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    train_loss += losses.item()

                avg_train_loss = train_loss/len(train_loader)
                val_loss = self.evaluate(model, val_loader)

                loss_history[fold]['train'].append(avg_train_loss)
                loss_history[fold]['val'].append(val_loss)

                print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f},',
                      f' Val Loss: {val_loss:.4f}')

            torch.save(
                model.state_dict(),
                os.path.join(self.model_dir, f"frcnn_fold{fold+1}.pth")
            )
            print(f"Saved model for fold {fold+1}")
            self.visualize_predictions(model, val_subset, label_map)
            self.plot_train_val_losses(loss_history, fold)


if __name__ == "__main__":
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    # Splits default to 0.70/0.15/0.15, set the splits as arguments
    check_annotations(root_path)
    train_path = os.path.join(root_path, 'train_val_annotations.json')
    dataset = TT100KFRCNNDataset(annotations_file=train_path, root_dir=root_path, transform=get_transform(train=True))
    num_classes = len(dataset.label_to_idx) + 1  # +1 for background class

    trainer = FasterRCNNKFoldTrainer(dataset, num_classes=num_classes, k_folds=5, epochs=5)
    trainer.train()
