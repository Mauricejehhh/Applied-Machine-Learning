import os
import json
from typing import Optional, Callable, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from PIL import Image, Image as PILImage
import torchvision.transforms.functional as F
import random


class TT100KFRCNNDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.labels = sorted(self.annotations["types"])
        self.label_to_idx = {label: idx + 1 for idx, label in enumerate(self.labels)}  # COCO starts from 1
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, img_data['path'])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes, labels = [], []
        for obj in img_data['objects']:
            bbox = obj['bbox']
            category = obj['category']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_to_idx[category])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)


class TT100KFRCNNDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.labels = sorted(self.annotations["types"])
        self.label_to_idx = {label: idx + 1 for idx, label in enumerate(self.labels)}  # COCO starts from 1
        self.transform = transform

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, img_data['path'])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes, labels = [], []
        for obj in img_data['objects']:
            bbox = obj['bbox']
            category = obj['category']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_to_idx[category])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)


class TT100KDataset(Dataset):
    """
    A dataset for loading full images and
    corresponding bounding box annotations
    from the TT100K dataset.
    """
    def __init__(self, annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            annotations_file (str): Path to the JSON annotation file.
            root_dir (str): Directory with all the images.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids: List[str] = list(self.annotations['imgs'].keys())
        self.root_dir: str = root_dir
        self.transform: Optional[Callable] = transform
        self.labels: List[str] = sorted(self.annotations["types"])
        self.label_to_idx: Dict[str, int] = {label: idx for idx,
                                             label in enumerate(self.labels)}
        self.idx_to_label: Dict[int, str] = {idx: label for label,
                                             idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image,
                                             Dict[str, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[Image, Dict[str, torch.Tensor]]: A tuple containing
                the image and its target dictionary,
                which includes bounding boxes and labels.
        """
        img_id = self.image_ids[idx]
        img_data = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, img_data['path'])
        image = Image.open(img_path).convert('RGB')

        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in img_data['objects']:
            bbox = obj['bbox']
            category = obj['category']

            w, h = image.size
            xmin = bbox['xmin'] / w
            ymin = bbox['ymin'] / h
            xmax = bbox['xmax'] / w
            ymax = bbox['ymax'] / h

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_to_idx[category])

        target = {
            'boxes': torch.tensor(boxes[0], dtype=torch.float32),
            'labels': torch.tensor(labels[0], dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


class TT100KSignDataset(Dataset):
    """
    A dataset for loading cropped traffic sign images from the TT100K dataset.
    """
    def __init__(self, annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            annotations_file (str): Path to the JSON annotation file.
            root_dir (str): Directory with all the images.
            transform (Callable, optional): Optional transform to be applied on a sample.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids: List[str] = list(self.annotations['imgs'].keys())
        self.root_dir: str = root_dir
        self.transform: Optional[Callable] = transform
        self.labels: List[str] = sorted(self.annotations["types"])
        self.label_to_idx: Dict[str, int] = {label: idx for idx,
                                             label in enumerate(self.labels)}
        self.idx_to_label: Dict[int, str] = {idx: label for idx,
                                             label in enumerate(self.labels)}
        self.data: List[Dict] = []

        for img_id, img_data in self.annotations['imgs'].items():
            img_path = os.path.join(self.root_dir, img_data['path'])
            for obj in img_data['objects']:
                bbox = obj['bbox']
                category = obj['category']

                self.data.append({
                    'img_path': img_path,
                    'bbox': bbox,
                    'category': category
                })

    def __len__(self) -> int:
        """Returns the total number of cropped signs."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, int]:
        """
        Args:
            idx (int): Index of the cropped image to retrieve.

        Returns:
            Tuple[Image, int]: A tuple containing the cropped image and its label index.
        """
        entry = self.data[idx]
        img_path = entry['img_path']
        bbox = entry["bbox"]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])
            cropped_image = image.crop((xmin, ymin, xmax, ymax))
            image = self.transform(cropped_image)

        return image, self.label_to_idx[entry['category']]
