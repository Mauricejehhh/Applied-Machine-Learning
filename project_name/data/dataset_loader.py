import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Any


class TK100Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.targets = []

        for fname in os.listdir(root):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                img_path = os.path.join(root, fname)
                ann_path = img_path.replace(".jpg", ".json").replace(".png", ".json")
                if os.path.exists(ann_path):
                    self.images.append(img_path)
                    with open(ann_path, "r") as f:
                        box = json.load(f)["bbox"]  # [x, y, w, h], normalized
                        self.targets.append(torch.tensor(box, dtype=torch.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target


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
