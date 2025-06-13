import os
import json
from typing import Optional, Callable, Tuple, Dict, List
import torch
from torch.utils.data import Dataset
from PIL import Image, Image as PILImage


class TT100KFRCNNDataset(Dataset):
    """
    A dataset for loading images and corresponding annotations for object detection
    in a format suitable for training Faster R-CNN models using the TT100K dataset.
    """
    def __init__(self,
                 annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            annotations_file (str): Path to the JSON annotation file.
            root_dir (str): Directory containing image files.
            transform (Callable, optional): A function/transform to apply to the images.
        """
        with open(annotations_file, 'r') as f:
            self.annotations: Dict = json.load(f)

        self.image_ids: List[str] = list(self.annotations['imgs'].keys())
        self.root_dir: str = root_dir
        self.labels: List[str] = sorted(self.annotations["types"])
        self.label_to_idx: Dict[str, int] = {label: idx + 1 for idx, label in enumerate(self.labels)}
        self.transform: Optional[Callable] = transform

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, Dict[str, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the image and its annotations to retrieve.

        Returns:
            Tuple[Image, Dict[str, torch.Tensor]]: A tuple containing:
                - The loaded image as a PIL Image.
                - A dictionary with:
                    - 'boxes': Tensor of bounding boxes (shape: [N, 4]).
                    - 'labels': Tensor of label indices.
                    - 'image_id': Tensor with the image index.
        """
        img_id = self.image_ids[idx]
        img_data = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, img_data['path'])
        image = Image.open(img_path).convert("RGB")

        boxes: List[List[float]] = []
        labels: List[int] = []

        for obj in img_data['objects']:
            bbox = obj['bbox']
            category = obj['category']
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_to_idx[category])

        target: Dict[str, torch.Tensor] = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            image = self.transform(image)

        return image, target


class TT100KDataset(Dataset):
    """
    A dataset for loading full images and normalized bounding box annotations
    from the TT100K dataset, suitable for classification or detection tasks.
    """
    def __init__(self,
                 annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            annotations_file (str): Path to the JSON annotation file.
            root_dir (str): Directory with all the images.
            transform (Callable, optional): Transform to be applied on a sample.
        """
        with open(annotations_file, 'r') as f:
            self.annotations: Dict = json.load(f)

        self.image_ids: List[str] = list(self.annotations['imgs'].keys())
        self.root_dir: str = root_dir
        self.transform: Optional[Callable] = transform
        self.labels: List[str] = sorted(self.annotations["types"])
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label: Dict[int, str] = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, Dict[str, torch.Tensor]]:
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[Image, Dict[str, torch.Tensor]]: A tuple containing:
                - The image.
                - A dictionary with 'boxes' and 'labels' tensors (first object only).
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

        target: Dict[str, torch.Tensor] = {
            'boxes': torch.tensor(boxes[0], dtype=torch.float32),
            'labels': torch.tensor(labels[0], dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


class TT100KSignDataset(Dataset):
    """
    A dataset for loading individual cropped traffic sign images from the TT100K dataset.
    Useful for training classification models on cropped traffic sign regions.
    """
    def __init__(self,
                 annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Args:
            annotations_file (str): Path to the JSON annotation file.
            root_dir (str): Directory containing the image files.
            transform (Callable, optional): A function/transform to apply to cropped images.
        """
        with open(annotations_file, 'r') as f:
            self.annotations: Dict = json.load(f)

        self.image_ids: List[str] = list(self.annotations['imgs'].keys())
        self.root_dir: str = root_dir
        self.transform: Optional[Callable] = transform
        self.labels: List[str] = sorted(self.annotations["types"])
        self.label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label: Dict[int, str] = {idx: label for idx, label in enumerate(self.labels)}
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
        """Returns the total number of cropped traffic sign entries."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[PILImage.Image, int]:
        """
        Args:
            idx (int): Index of the cropped image to retrieve.

        Returns:
            Tuple[Image, int]: A tuple with:
                - Cropped PIL image of the traffic sign.
                - Integer label index for the traffic sign.
        """
        entry = self.data[idx]
        img_path = entry['img_path']
        bbox = entry["bbox"]
        image = Image.open(img_path).convert('RGB')

        xmin = int(bbox["xmin"])
        ymin = int(bbox["ymin"])
        xmax = int(bbox["xmax"])
        ymax = int(bbox["ymax"])
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, self.label_to_idx[entry['category']]
