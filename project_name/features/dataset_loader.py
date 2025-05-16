import os
from torch.utils.data import Dataset
import json
from PIL import Image


class TT100KDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['imgs'])

    def __getitem__(self, idx):
        entry = self.annotations['imgs'].values()[idx]
        img_path = os.path.join(self.root_dir, entry['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
