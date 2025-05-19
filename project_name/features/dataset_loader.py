import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from .preprocessing import preprocess_image, preprocess_and_crop_image


class TT100KDataset(Dataset):
    """ Torch Dataset Class for the original TT100K data.
    Images returns from __getitem__() are the grayscaled 512x512 images.
    """
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in
                             enumerate(self.annotations['types'])}

    def __len__(self):
        return len(self.annotations['imgs'])

    def __getitem__(self, idx):
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
        self.label_map = {label: idx for idx, label in enumerate(self.labels)}
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
        image = preprocess_and_crop_image(image, bbox)
        image = torch.Tensor(image)

        if self.transform:
            image = self.transform(image)
        return image, self.label_map[entry['category']]
