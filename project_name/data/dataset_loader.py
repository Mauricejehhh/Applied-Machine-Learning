import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class TT100KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(self.annotations["types"])
        self.label_to_idx = {label: idx for idx,
                             label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label,
                             idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, img_data['path'])
        image = Image.open(img_path).convert('RGB')

        obj = img_data['objects'][0]
        bbox = obj['bbox']
        category = obj['category']

        w, h = image.size
        xmin = bbox['xmin'] / w
        ymin = bbox['ymin'] / h
        xmax = bbox['xmax'] / w
        ymax = bbox['ymax'] / h

        boxes = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        target = {
            'boxes': boxes,  # Shape: [4]
            'labels': torch.tensor(self.label_to_idx[category])
        }

        if self.transform:
            image = self.transform(image)

        return image, target


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
