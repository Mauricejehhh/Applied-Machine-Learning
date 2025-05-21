import os
import json
from torch.utils.data import Dataset
from .preprocessing import preprocess_image


class TT100KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations['imgs'].keys())
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['imgs'])

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        entry = self.annotations['imgs'][img_id]
        img_path = os.path.join(self.root_dir, entry['path'])
        image = preprocess_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
