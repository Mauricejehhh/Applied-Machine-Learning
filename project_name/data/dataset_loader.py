import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Any
from .dataset_visualizer import preprocess_image


class TT100KDataset(Dataset):
    """Custom dataset for the TT100K traffic sign dataset."""

    def __init__(self,
                 annotations_file: str,
                 root_dir: str,
                 transform: Optional[Callable] = None):
        """
        Initializes the TT100KDataset.

        Args:
            annotations_file (str): Path to the JSON file with annotations.
            root_dir (str): Directory with all the images.
            transform (Optional[Callable], optional): Optional transform to be
            applied on a sample. Defaults to None.
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in
                             enumerate(self.annotations['types'])}

    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.annotations['imgs'])

    def __getitem__(self, idx: int) -> Any:
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Any: Transformed image sample.
        """
        img_id = self.image_ids[idx]
        entry = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, entry['path'])
        image = Image.open(img_path).convert('RGB')
        image = preprocess_image(image)
        image = torch.Tensor(image)
        labels = [self.class_to_idx[obj['category']]
                  for obj in entry['objects']]
        label_tensor = torch.zeros(len(self.class_to_idx))
        label_tensor[labels] = 1
        if self.transform:
            image = self.transform(image)
        return image, label_tensor


class TT100KSignDataset(Dataset):
    """ Dataset that contains the cropped sign images,
    instead of the entire picture.
    """
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(self.annotations["types"])
        self.label_to_idx = {label: idx for idx,
                             label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for idx,
                             label in enumerate(self.labels)}
        self.data = []

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
